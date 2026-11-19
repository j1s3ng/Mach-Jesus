import math

import pytest

from race_lidar.scored_pipeline import wrap_scored_predictor


def pipeline():
    return {"version": 1, "effective_car_score_threshold": .35, "threshold_selection": "fixed_predeclared",
            "score_calibration_sha256": None, "weights_parent_checkpoint_sha256": "parent", "weights_state_sha256": "state"}


def test_health_attests_actual_inclusive_score_floor_without_mixing_independent_cars():
    rows = [[{"score": .3499}, {"score": .35}, {"score": .8}], [{"score": .2}]]
    def detector(clouds): return rows
    detector.metadata = {"checkpoint_sha256": "artifact"}
    wrapped = wrap_scored_predictor(detector, pipeline())
    assert wrapped([None, None]) == [[rows[0][1], rows[0][2]], []]
    assert wrapped.metadata["checkpoint_sha256"] == "artifact"
    assert wrapped.metadata["effective_car_score_threshold"] == .35
    assert wrapped.metadata["score_calibration_sha256"] is None
    assert rows[0][0]["score"] == .3499


def test_invalid_score_or_wrong_batch_length_fails():
    with pytest.raises(ValueError): wrap_scored_predictor(lambda clouds: [], {**pipeline(), "effective_car_score_threshold": math.nan})
    wrapped = wrap_scored_predictor(lambda clouds: [], pipeline())
    with pytest.raises(RuntimeError): wrapped([None])
