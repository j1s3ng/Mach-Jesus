import copy
import numpy as np

from race_lidar.rich_driver import FEATURE_NAMES, save_rich_model
from scripts.prepare_finalist_selection import numerical_model_identity, cem_validation_gate, proposed_cem_selection_jobs


def model(path, *, seed=7, mode="stopping_critical", adjustment=0.):
    weight = np.zeros((6, len(FEATURE_NAMES)), dtype=np.float32)
    weight[0, 0] = adjustment
    save_rich_model(path, [weight], [np.zeros(6)], np.zeros(len(FEATURE_NAMES)),
                    np.ones(len(FEATURE_NAMES)), {"seed": seed, "lane_summary_mode": mode})
    return path


def completed_state():
    winner = {"candidate_id": "winner", "parameters": [1.02, 0., 0., 0., 0.]}
    return {"status": "complete", "reason": "development_gain_verified", "winner": winner,
        "generations": [{"index": i, "completed": True, "candidates": [{"reward": {}} for _ in range(6)]} for i in range(2)],
        "validation": {"seed": 211, "rotation": 1, "used_for_cem_updates": False,
            "development_improvement": True, "winner": winner,
            "results": {"baseline": {"passed": True, "mean_finish_time_s": 250.},
                        "winner": {"passed": True, "mean_finish_time_s": 249.}}}}


def test_metadata_and_zip_sha_changes_cannot_create_an_extra_racer(tmp_path):
    a = numerical_model_identity(model(tmp_path/"a.npz", seed=7))
    b = numerical_model_identity(model(tmp_path/"b.npz", seed=17))
    assert a["checkpoint_sha256"] != b["checkpoint_sha256"]
    assert a["numerical_model_sha256"] == b["numerical_model_sha256"]


def test_real_numeric_weights_or_input_encoding_change_identity(tmp_path):
    baseline = numerical_model_identity(model(tmp_path/"a.npz"))["numerical_model_sha256"]
    changed = numerical_model_identity(model(tmp_path/"b.npz", adjustment=.01))["numerical_model_sha256"]
    encoding = numerical_model_identity(model(tmp_path/"c.npz", mode="nearest"))["numerical_model_sha256"]
    assert baseline != changed and baseline != encoding


def test_callback_never_prepares_jobs_for_identity_missing_fitness_or_unvalidated_cem():
    state = completed_state()
    assert cem_validation_gate(state) == []
    identity = copy.deepcopy(state)
    identity["winner"]["parameters"] = [1., 0., 0., 0., 0.]
    assert "winner_missing_or_identity_calibration" in cem_validation_gate(identity)
    state["generations"][1]["candidates"][0].pop("reward")
    assert "all_twelve_fitness_records_not_preserved" in cem_validation_gate(state)
    prepared = proposed_cem_selection_jobs(state)
    assert not prepared["ready"] and prepared["jobs"] == []


def test_seed211_validation_is_independent_clean_and_at_least_one_second():
    for change in ("gain", "seed", "feedback", "unclean"):
        state = completed_state()
        if change == "gain": state["validation"]["results"]["winner"]["mean_finish_time_s"] = 249.01
        if change == "seed": state["validation"]["seed"] = 101
        if change == "feedback": state["validation"]["used_for_cem_updates"] = True
        if change == "unclean": state["validation"]["results"]["winner"]["passed"] = False
        assert cem_validation_gate(state)
