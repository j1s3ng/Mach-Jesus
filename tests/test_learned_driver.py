import numpy as np
import pytest

from race_lidar.learned_driver import FEATURE_NAMES, LearnedDriver, feature_vector, save_numpy_model


def state():
    return {'progress_fraction':.2,'body_speed_mps':30.,'curvature_ahead':[0,.001,.005,.02,.01,0,0],
            'front_distance_m':30.,'front_speed_mps':12.,'left_clearance_m':10.,'right_clearance_m':1.}


def test_policy_inputs_use_perception_but_exclude_ground_truth_and_teacher_speed():
    original=feature_vector(state())
    extra={**state(),'reference_speed_mps':999.,'gt_boxes':[{'center':[1,2,3]}]}
    np.testing.assert_array_equal(original,feature_vector(extra))
    changed=feature_vector({**state(),'front_distance_m':8.})
    assert not np.array_equal(original,changed)
    assert changed[FEATURE_NAMES.index('front_distance')]==pytest.approx(8/60)


def test_numpy_export_matches_torch_network(tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(12)
    model=torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES),16),torch.nn.ReLU(),torch.nn.Linear(16,2))
    mean=np.full(len(FEATURE_NAMES),.1,np.float32);scale=np.full(len(FEATURE_NAMES),.8,np.float32)
    layers=[model[0],model[2]]
    path=save_numpy_model(tmp_path/'model.npz',[layer.weight.detach().numpy() for layer in layers],
                          [layer.bias.detach().numpy() for layer in layers],mean,scale,{'trained':True})
    portable=LearnedDriver.from_npz(path)
    x=feature_vector(state())
    with torch.inference_mode():expected=model(torch.from_numpy((x-mean)/scale)).numpy()
    np.testing.assert_allclose(portable.forward_features(x),expected,atol=1e-6)
    assert portable.checkpoint_sha256 and portable.metadata['trained'] is True


def test_outputs_are_bounded_and_malformed_features_rejected():
    model=LearnedDriver([np.zeros((2,len(FEATURE_NAMES)))],[np.array([10.,-10.])],
                         np.zeros(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES)))
    prediction=model.predict(state())
    assert prediction=={'target_speed_mps':60.,'target_offset_m':-3.5,'model_used':True}
    with pytest.raises(ValueError,match='seven finite'):
        model.predict({**state(),'curvature_ahead':[0,1]})
    with pytest.raises(ValueError,match='finite'):
        model.predict({**state(),'body_speed_mps':float('nan')})
