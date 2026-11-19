from scripts.calibrate_detector_score import select_threshold


def examples(false_score, true_score=.8):
    box = {"center": [12., 0., .5], "size": [4., 2., 1.], "yaw": 0.}
    truth, official, learned = {}, [], []
    for i in range(100):
        key = {"scene_id": "train", "frame_id": str(i)}
        truth[("train", str(i))] = {"gt_boxes": [box]}
        official.append({**key, "boxes": [{**box, "score": .7}]})
        boxes = [{**box, "score": true_score}]
        if i < 2: boxes.append({**box, "center": [40., 0., .5], "score": false_score})
        learned.append({**key, "boxes": boxes})
    return official, learned, truth


PROTOCOL = {"official_baseline_score_threshold": .35, "matching_distance_m": 2.,
            "raw_score_threshold_grid": [round(.35+i*.025, 3) for i in range(25)],
            "objective": {"maximum_false_positives_per_training_frame": .01}}


def test_grid_selects_most_recall_then_lowest_threshold_that_meets_fp_budget():
    result = select_threshold(*examples(.36), PROTOCOL)
    assert result["selected"]["effective_raw_score_threshold"] == .375
    assert result["selected"]["recall"] == 1.
    assert result["new_functional_candidate"]


def test_unchanged_threshold_is_explicitly_not_a_new_candidate():
    result = select_threshold(*examples(.3), PROTOCOL)
    assert result["selected"]["effective_raw_score_threshold"] == .35
    assert not result["new_functional_candidate"]


def test_no_eligible_rule_does_not_relax_fp_or_official_recall_constraints():
    result = select_threshold(*examples(.95, true_score=.6), PROTOCOL)
    assert result["selected"] is None
    assert not result["new_functional_candidate"]
