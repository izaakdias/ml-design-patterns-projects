"""Pretrain on one synthetic line pattern, adapt, checkpoint, and quantize."""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ml_patterns_lab.shared import save_bundle
from ml_patterns_lab.vision.cascade import route_image
from ml_patterns_lab.vision.data import IMAGE_SIZE, make_images, quantize_images


def main() -> None:
    source_images, source_labels = make_images(rows=1000, domain="source", seed=51)
    source_pixels = quantize_images(source_images).reshape(-1, IMAGE_SIZE)
    fast_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.002,
        batch_size=64,
        learning_rate_init=0.002,
        max_iter=1,
        random_state=51,
    )
    for epoch in range(5):
        if epoch == 0:
            fast_model.partial_fit(source_pixels, source_labels, classes=np.array([0, 1]))
        else:
            fast_model.partial_fit(source_pixels, source_labels)
        save_bundle(
            "vision",
            {"fast": fast_model},
            {
                "phase": "source_pretraining",
                "epoch": epoch + 1,
                "source_domain": "vertical scratches",
                "input_quantization_bits": 5,
            },
        )

    target_images, target_labels = make_images(rows=1600, domain="target", seed=52)
    images_train_valid, images_test, labels_train_valid, labels_test = train_test_split(
        target_images,
        target_labels,
        test_size=0.20,
        random_state=52,
        stratify=target_labels,
    )
    images_train, images_validation, labels_train, labels_validation = train_test_split(
        images_train_valid,
        labels_train_valid,
        test_size=0.25,
        random_state=53,
        stratify=labels_train_valid,
    )
    train_pixels = quantize_images(images_train).reshape(-1, IMAGE_SIZE)
    validation_pixels = quantize_images(images_validation).reshape(-1, IMAGE_SIZE)
    test_pixels = quantize_images(images_test).reshape(-1, IMAGE_SIZE)

    for epoch in range(5):
        fast_model.partial_fit(train_pixels, labels_train)
        save_bundle(
            "vision",
            {"fast": fast_model},
            {
                "phase": "target_fine_tuning",
                "epoch": epoch + 1,
                "source_domain": "vertical scratches",
                "target_domain": "diagonal scratches and changed lighting",
                "input_quantization_bits": 5,
            },
        )

    confidence = fast_model.predict_proba(validation_pixels).max(axis=1)
    fast_threshold = float(np.quantile(confidence, 0.35))
    expert_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=54,
    )
    expert_model.fit(train_pixels, labels_train)
    bundle = {"fast": fast_model, "expert": expert_model}

    quantized_predictions = fast_model.predict(test_pixels)
    raw_test_pixels = images_test.reshape(-1, IMAGE_SIZE)
    raw_predictions = fast_model.predict(raw_test_pixels)
    routed = [
        route_image(bundle, image.reshape(-1), fast_threshold, 0.65)
        for image in test_pixels
    ]
    handled = [item["inspection"] != "manual_review" for item in routed]
    handled_count = max(sum(handled), 1)
    handled_accuracy = sum(
        (item["inspection"] == "defect") == bool(label)
        for item, label, is_handled in zip(routed, labels_test, handled)
        if is_handled
    ) / handled_count
    metrics = {
        "fast_accuracy_with_quantized_pixels": round(
            float(accuracy_score(labels_test, quantized_predictions)),
            4,
        ),
        "fast_accuracy_with_full_precision_pixels": round(
            float(accuracy_score(labels_test, raw_predictions)),
            4,
        ),
        "cascade_coverage": round(float(np.mean(handled)), 4),
        "cascade_accuracy_when_handled": round(float(handled_accuracy), 4),
        "manual_review_share": round(
            float(sum(item["manual_review_required"] for item in routed) / len(routed)),
            4,
        ),
        "input_quantization_bits": 5,
    }
    version_path, metadata = save_bundle(
        "vision",
        bundle,
        {
            "image_shape": [24, 24],
            "fast_confidence_threshold": fast_threshold,
            "expert_review_threshold": 0.65,
            "metrics": metrics,
            "source_pretraining_rows": int(len(source_labels)),
            "target_fine_tuning_rows": int(len(labels_train)),
        },
    )
    print(json.dumps({"artifact": str(version_path), "metadata": metadata}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
