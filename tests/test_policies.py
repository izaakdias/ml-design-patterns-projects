import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from ml_patterns_lab.evaluation import (
    select_confidence_threshold,
    select_recall_threshold,
    slice_metrics,
)
from ml_patterns_lab.fraud.model import choose_urgent_threshold, review_action
from ml_patterns_lab.support.cascade import route_ticket
from ml_patterns_lab.vision.cascade import route_image
from ml_patterns_lab.vision.data import quantize_images


def test_recall_policy_searches_actual_scores_below_old_grid():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.001, 0.002, 0.0001, 0.003])
    threshold = select_recall_threshold(labels, scores, 1.0, objective="threshold")
    assert threshold == 0.001
    assert (scores[labels == 1] >= threshold).mean() == 1


@pytest.mark.parametrize("scores", [[float("nan"), 0.5], [-1, 0.5], [0.5, 2]])
def test_nonfinite_or_invalid_probabilities_rejected(scores):
    with pytest.raises(ValueError):
        select_recall_threshold(np.array([1, 0]), np.array(scores))


def test_urgent_band_requires_support_and_can_be_disabled():
    labels = np.array([1, 0, 0])
    scores = np.array([0.99, 0.9, 0.8])
    assert choose_urgent_threshold(labels, scores, 0.2) is None
    assert review_action(0.99, 0.2, None) == "human_review"
    assert review_action(0.95, 0.2, 0.9) == "urgent_human_review"
    assert review_action(0.1, 0.2, None) == "standard_queue"


def test_confidence_routing_can_abstain_instead_of_fabricating_threshold():
    labels = np.array([0, 0, 0, 0, 0])
    predictions = np.ones(5)
    assert (
        select_confidence_threshold(labels, predictions, np.ones(5), minimum_accuracy=0.9) is None
    )


def test_missing_slice_class_is_reported_as_undefined():
    metrics = slice_metrics(np.array(["a", "a"]), np.array([0, 0]), np.array([0, 1]))
    assert metrics["a"]["recall"] is None
    assert metrics["a"]["false_positive_rate"] == 0.5


class FixedClassifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.classes_ = np.array(["cobranca", "tecnico"])
        vectorizer = TfidfVectorizer().fit(["cobranca tecnico"])
        self.named_steps = {"text_features": vectorizer}

    def predict_proba(self, rows):
        return np.array([self.probabilities for _ in rows])


@pytest.mark.parametrize(
    ("fast", "expert", "stage"),
    [
        ([0.95, 0.05], [0.5, 0.5], "fast"),
        ([0.5, 0.5], [0.1, 0.9], "expert"),
        ([0.5, 0.5], [0.5, 0.5], "neutral"),
    ],
)
def test_support_cascade_routes_each_branch(fast, expert, stage):
    result = route_ticket(FixedClassifier(fast), FixedClassifier(expert), "cobranca", 0.8, 0.6)
    assert result["stage"] == stage
    assert result["human_review_required"] == (stage == "neutral")


def test_unknown_vocabulary_goes_to_review():
    result = route_ticket(
        FixedClassifier([0.99, 0.01]), FixedClassifier([0.99, 0.01]), "xyzzy", 0.8
    )
    assert result["human_review_required"]


@pytest.mark.parametrize(
    ("fast", "expert", "stage"),
    [
        ([0.95, 0.05], [0.5, 0.5], "fast"),
        ([0.5, 0.5], [0.1, 0.9], "expert"),
        ([0.5, 0.5], [0.5, 0.5], "review"),
    ],
)
def test_vision_cascade_routes_each_branch(fast, expert, stage):
    result = route_image(
        {"fast": FixedClassifier(fast), "expert": FixedClassifier(expert)}, [0.2] * 576, 0.8, 0.6
    )
    assert result["stage"] == stage


@pytest.mark.parametrize("bits", [0, 17, 2.5])
def test_quantization_rejects_invalid_bit_depth(bits):
    with pytest.raises(ValueError):
        quantize_images(np.array([0.1]), bits)


def test_quantization_is_idempotent_and_bounded():
    quantized = quantize_images(np.linspace(0, 1, 100))
    np.testing.assert_array_equal(quantize_images(quantized), quantized)
    assert 0 <= quantized.min() <= quantized.max() <= 1
