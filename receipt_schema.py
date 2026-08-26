"""Schema, normalisation and metrics for receipt -> JSON extraction.

This module holds every piece of logic that does NOT need a GPU, which is why
it lives outside the notebook: it can be unit-tested in CI on a plain runner.

Three jobs:
  1. Define the target schema (Pydantic) that the model must produce.
  2. Turn CORD's raw ground truth into that schema, so training targets and
     model outputs are directly comparable.
  3. Score a prediction against the gold answer, field by field.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_serializer

# --------------------------------------------------------------------------
# Schema - this is the contract the fine-tuned model learns to emit
# --------------------------------------------------------------------------


def _compact(value: Optional[float]) -> Optional[float | int]:
    """Emit 10000 rather than 10000.0 so training targets stay clean."""
    if value is None:
        return None
    return int(value) if float(value).is_integer() else round(float(value), 2)


class LineItem(BaseModel):
    name: str
    quantity: Optional[float] = None
    price: Optional[float] = None

    @field_serializer("quantity", "price")
    def _ser(self, v: Optional[float]) -> Optional[float | int]:
        return _compact(v)


class Receipt(BaseModel):
    items: List[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    total: Optional[float] = None

    @field_serializer("subtotal", "total")
    def _ser(self, v: Optional[float]) -> Optional[float | int]:
        return _compact(v)

    def to_json(self) -> str:
        """Canonical, deterministic JSON string - used as the training target."""
        return json.dumps(self.model_dump(), ensure_ascii=False)


#: Shown to the model in the prompt so it knows the expected shape.
SCHEMA_HINT = (
    '{"items": [{"name": str, "quantity": number|null, "price": number|null}], '
    '"subtotal": number|null, "total": number|null}'
)

INSTRUCTION = (
    "Extract the receipt into JSON matching this schema:\n"
    f"{SCHEMA_HINT}\n"
    "Return only the JSON object, with no explanation and no markdown fences."
)


# --------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------

_NUM_CHARS = re.compile(r"[^0-9.,\-]")


def parse_money(value: Any) -> Optional[float]:
    """Parse a price string into a float.

    Receipt prices are written inconsistently, and CORD is Indonesian, where
    '10.000' means ten thousand rather than ten. Rules applied in order:
      - both ',' and '.' present -> the RIGHTMOST one is the decimal separator
      - one separator repeated   -> it is a thousands separator
      - one separator once       -> thousands if exactly 3 digits follow, else decimal
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = _NUM_CHARS.sub("", str(value)).strip()
    if not text or text in {"-", ".", ","}:
        return None

    negative = text.startswith("-")
    text = text.lstrip("-")

    has_comma, has_dot = "," in text, "." in text

    if has_comma and has_dot:
        decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
        thousands_sep = "." if decimal_sep == "," else ","
        text = text.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_comma or has_dot:
        sep = "," if has_comma else "."
        if text.count(sep) > 1:
            text = text.replace(sep, "")
        else:
            head, _, tail = text.partition(sep)
            # exactly three trailing digits -> thousands grouping (10.000)
            text = head + tail if len(tail) == 3 and tail.isdigit() else head + "." + tail

    try:
        number = float(text)
    except ValueError:
        return None
    return -number if negative else number


def parse_quantity(value: Any) -> Optional[float]:
    """'2 x' -> 2.0, 'x3' -> 3.0, '1' -> 1.0."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:[.,]\d+)?", str(value))
    return parse_money(match.group()) if match else None


def normalize_name(name: str) -> str:
    """Lowercase and strip punctuation so item names can be compared fairly."""
    return re.sub(r"[^a-z0-9 ]", "", str(name).lower()).strip()


def extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of raw model output.

    Models wrap JSON in prose or ```json fences, especially before fine-tuning,
    so scan for the first balanced {...} instead of trusting the whole string.
    """
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)

    start = text.find("{")
    if start == -1:
        return None

    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# --------------------------------------------------------------------------
# CORD ground truth -> our schema
# --------------------------------------------------------------------------


def _as_list(value: Any) -> List[dict]:
    """CORD stores a single menu row as a dict and multiple rows as a list."""
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    return [v for v in value if isinstance(v, dict)]


def _first_str(value: Any) -> str:
    """A CORD 'nm' field is occasionally a list of strings."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value) if value is not None else ""


def normalize_cord(ground_truth: Any) -> Receipt:
    """Convert one CORD record's ground truth into a Receipt.

    Accepts either the raw JSON string from the dataset or an already-parsed
    dict, with or without the outer 'gt_parse' wrapper.
    """
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return Receipt()
    if not isinstance(ground_truth, dict):
        return Receipt()

    parsed = ground_truth.get("gt_parse", ground_truth)
    if not isinstance(parsed, dict):
        return Receipt()

    items: List[LineItem] = []
    for row in _as_list(parsed.get("menu")):
        name = _first_str(row.get("nm")).strip()
        if not name:
            continue
        items.append(
            LineItem(
                name=name,
                quantity=parse_quantity(row.get("cnt")),
                price=parse_money(row.get("price")),
            )
        )

    sub_total = parsed.get("sub_total") or {}
    total = parsed.get("total") or {}
    if not isinstance(sub_total, dict):
        sub_total = {}
    if not isinstance(total, dict):
        total = {}

    return Receipt(
        items=items,
        subtotal=parse_money(sub_total.get("subtotal_price")),
        total=parse_money(total.get("total_price")),
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def _values_match(a: Optional[float], b: Optional[float], tol: float = 0.01) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def score_prediction(raw_output: str, gold: Receipt) -> Dict[str, Any]:
    """Score one model output against the gold receipt.

    Reported per field, because a single accuracy number hides that totals are
    easy and line items are hard - which is the whole point of error analysis.
    """
    result: Dict[str, Any] = {
        "json_valid": False,
        "schema_valid": False,
        "total_correct": False,
        "subtotal_correct": False,
        "item_count_correct": False,
        "item_name_f1": 0.0,
        "item_price_accuracy": 0.0,
    }

    payload = extract_json(raw_output)
    if payload is None:
        return result
    result["json_valid"] = True

    try:
        prediction = Receipt.model_validate(payload)
    except Exception:
        return result
    result["schema_valid"] = True

    result["total_correct"] = _values_match(prediction.total, gold.total)
    result["subtotal_correct"] = _values_match(prediction.subtotal, gold.subtotal)
    result["item_count_correct"] = len(prediction.items) == len(gold.items)

    predicted_names = [normalize_name(i.name) for i in prediction.items]
    gold_names = [normalize_name(i.name) for i in gold.items]

    remaining = list(gold_names)
    overlap = 0
    for name in predicted_names:
        if name in remaining:
            remaining.remove(name)
            overlap += 1

    if predicted_names or gold_names:
        precision = overlap / len(predicted_names) if predicted_names else 0.0
        recall = overlap / len(gold_names) if gold_names else 0.0
        result["item_name_f1"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    else:
        result["item_name_f1"] = 1.0

    # Price accuracy is only meaningful for items whose name was matched.
    gold_by_name = {normalize_name(i.name): i.price for i in gold.items}
    matched = correct = 0
    for item in prediction.items:
        key = normalize_name(item.name)
        if key in gold_by_name:
            matched += 1
            if _values_match(item.price, gold_by_name[key]):
                correct += 1
    result["item_price_accuracy"] = correct / matched if matched else 0.0

    return result


def aggregate(scores: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of each metric across the evaluation set."""
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: sum(float(s[k]) for s in scores) / len(scores) for k in keys}
