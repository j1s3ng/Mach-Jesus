"""An explicit effective car-score floor bound into a detector artifact."""
from __future__ import annotations

import math


def wrap_scored_predictor(predict_batch, pipeline):
    threshold = pipeline["effective_car_score_threshold"]
    if pipeline["version"] != 1 or isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid effective detector score configuration")
    def scored(point_clouds):
        predictions = predict_batch(point_clouds)
        if len(predictions) != len(point_clouds): raise RuntimeError("Independent detector batch length differs")
        return [[box for box in boxes if box["score"] >= threshold] for boxes in predictions]
    scored.metadata = {**getattr(predict_batch, "metadata", {}),
                       "effective_car_score_threshold": threshold,
                       "score_threshold_selection": pipeline["threshold_selection"],
                       "score_calibration_sha256": pipeline["score_calibration_sha256"],
                       "weights_parent_checkpoint_sha256": pipeline["weights_parent_checkpoint_sha256"],
                       "weights_state_sha256": pipeline["weights_state_sha256"],
                       "pipeline": pipeline}
    if hasattr(predict_batch, "telemetry"): scored.telemetry = predict_batch.telemetry
    return scored
