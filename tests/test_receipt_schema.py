"""Tests for the GPU-free logic: parsing, normalisation and scoring.

These run on a plain CI runner in seconds - no model, no GPU, no dataset.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from receipt_schema import (  # noqa: E402
    LineItem,
    Receipt,
    aggregate,
    extract_json,
    normalize_cord,
    normalize_name,
    parse_money,
    parse_quantity,
    score_prediction,
)


# ----------------------------------------------------------------- parse_money


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10,000", 10000.0),      # comma thousands
        ("10.000", 10000.0),      # dot thousands (Indonesian - CORD)
        ("1,234.56", 1234.56),    # US style
        ("12.345,67", 12345.67),  # European style
        ("1.234.567", 1234567.0), # repeated separator
        ("Rp 25,000", 25000.0),   # currency prefix
        ("3,50", 3.50),           # decimal comma, 2 dp
        ("100", 100.0),
        (5000, 5000.0),
        (12.5, 12.5),
        ("-1,500", -1500.0),
        ("", None),
        (None, None),
        ("n/a", None),
    ],
)
def test_parse_money(raw, expected):
    assert parse_money(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [("2 x", 2.0), ("x3", 3.0), ("1", 1.0), (4, 4.0), (None, None), ("", None)],
)
def test_parse_quantity(raw, expected):
    assert parse_quantity(raw) == expected


def test_normalize_name_strips_punctuation_and_case():
    assert normalize_name("  Nasi-Goreng (Spicy)! ") == "nasigoreng spicy"


# ---------------------------------------------------------------- extract_json


def test_extract_json_plain():
    assert extract_json('{"total": 5}') == {"total": 5}


def test_extract_json_from_markdown_fence():
    text = 'Here you go:\n```json\n{"total": 12}\n```\nHope that helps!'
    assert extract_json(text) == {"total": 12}


def test_extract_json_ignores_trailing_prose():
    text = '{"items": [], "total": 1} and that is the answer.'
    assert extract_json(text) == {"items": [], "total": 1}


def test_extract_json_handles_nested_and_braces_in_strings():
    text = 'noise {"items": [{"name": "a}b", "price": 2}], "total": 2} more noise'
    assert extract_json(text) == {"items": [{"name": "a}b", "price": 2}], "total": 2}


def test_extract_json_returns_none_when_absent_or_broken():
    assert extract_json("no json at all") is None
    assert extract_json('{"total": }') is None
    assert extract_json("") is None


# --------------------------------------------------------------- normalize_cord

CORD_SAMPLE = json.dumps(
    {
        "gt_parse": {
            "menu": [
                {"nm": "NASI GORENG", "cnt": "2 x", "price": "40,000"},
                {"nm": "ES TEH", "cnt": "1 x", "price": "8,000"},
            ],
            "sub_total": {"subtotal_price": "48,000"},
            "total": {"total_price": "48,000"},
        }
    }
)


def test_normalize_cord_parses_items_and_totals():
    receipt = normalize_cord(CORD_SAMPLE)
    assert len(receipt.items) == 2
    assert receipt.items[0].name == "NASI GORENG"
    assert receipt.items[0].quantity == 2.0
    assert receipt.items[0].price == 40000.0
    assert receipt.subtotal == 48000.0
    assert receipt.total == 48000.0


def test_normalize_cord_accepts_single_menu_dict():
    raw = {"gt_parse": {"menu": {"nm": "KOPI", "cnt": "1 x", "price": "15,000"}}}
    receipt = normalize_cord(raw)
    assert len(receipt.items) == 1
    assert receipt.items[0].name == "KOPI"


def test_normalize_cord_joins_list_valued_name():
    raw = {"gt_parse": {"menu": [{"nm": ["AYAM", "BAKAR"], "price": "30,000"}]}}
    assert normalize_cord(raw).items[0].name == "AYAM BAKAR"


def test_normalize_cord_skips_nameless_rows():
    raw = {"gt_parse": {"menu": [{"nm": "", "price": "1,000"}, {"nm": "TEH"}]}}
    assert [i.name for i in normalize_cord(raw).items] == ["TEH"]


def test_normalize_cord_survives_garbage():
    assert normalize_cord("not json").items == []
    assert normalize_cord(None).total is None
    assert normalize_cord({"gt_parse": {"total": "unexpected string"}}).total is None


def test_to_json_is_compact_and_roundtrips():
    receipt = normalize_cord(CORD_SAMPLE)
    payload = json.loads(receipt.to_json())
    # 40000 not 40000.0 - clean integers make easier training targets
    assert payload["items"][0]["price"] == 40000
    assert "." not in receipt.to_json().split('"price":')[1][:8]


# ------------------------------------------------------------ score_prediction


@pytest.fixture
def gold():
    return Receipt(
        items=[
            LineItem(name="NASI GORENG", quantity=2, price=40000),
            LineItem(name="ES TEH", quantity=1, price=8000),
        ],
        subtotal=48000,
        total=48000,
    )


def test_score_perfect_prediction(gold):
    score = score_prediction(gold.to_json(), gold)
    assert score["json_valid"] and score["schema_valid"]
    assert score["total_correct"] and score["subtotal_correct"]
    assert score["item_count_correct"]
    assert score["item_name_f1"] == 1.0
    assert score["item_price_accuracy"] == 1.0


def test_score_invalid_json_scores_zero(gold):
    score = score_prediction("I think the total is about 48,000 rupiah.", gold)
    assert score["json_valid"] is False
    assert score["item_name_f1"] == 0.0


def test_score_valid_json_wrong_values(gold):
    score = score_prediction('{"items": [], "subtotal": 1, "total": 2}', gold)
    assert score["json_valid"] and score["schema_valid"]
    assert not score["total_correct"]
    assert not score["item_count_correct"]


def test_score_partial_item_match(gold):
    prediction = '{"items": [{"name": "nasi goreng", "price": 40000}], "total": 48000}'
    score = score_prediction(prediction, gold)
    assert score["total_correct"]
    # one of two golds recovered -> precision 1.0, recall 0.5, F1 ~0.667
    assert score["item_name_f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert score["item_price_accuracy"] == 1.0


def test_score_rewards_json_in_prose(gold):
    """Base models wrap JSON in chatter; that should still count as valid."""
    noisy = f"Sure! Here is the extracted data:\n```json\n{gold.to_json()}\n```"
    assert score_prediction(noisy, gold)["total_correct"]


def test_empty_receipts_match(gold):
    empty = Receipt()
    assert score_prediction(empty.to_json(), empty)["item_name_f1"] == 1.0


def test_aggregate_averages_each_metric():
    scores = [
        {"json_valid": True, "total_correct": True},
        {"json_valid": True, "total_correct": False},
    ]
    summary = aggregate(scores)
    assert summary["json_valid"] == 1.0
    assert summary["total_correct"] == 0.5
    assert aggregate([]) == {}
