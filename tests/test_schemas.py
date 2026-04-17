"""Tests pour les schémas Pydantic."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "api"))


def test_recommend_item_valid():
    from schemas import RecommendItem

    item = RecommendItem(product_id="P001", score=0.85, source="als", category="Beauty")
    assert item.product_id == "P001"
    assert item.score == 0.85


def test_feedback_request_valid():
    from schemas import FeedbackRequest

    fb = FeedbackRequest(
        user_id="U001",
        product_id="P001",
        interaction_type="click",
        ab_variant="control",
    )
    assert fb.interaction_type == "click"
    assert fb.timestamp is None


def test_feedback_request_invalid_type():
    from schemas import FeedbackRequest

    with pytest.raises(Exception):
        FeedbackRequest(
            user_id="U001",
            product_id="P001",
            interaction_type="invalid_type",
            ab_variant="control",
        )
