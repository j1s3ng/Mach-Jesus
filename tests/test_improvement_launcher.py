"""A frozen race must build its world from that source, despite live imports."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('split',['cv_qualification','championship','geometric-confirmation'])
def test_prepare_uses_isolated_frozen_geometry_and_seals_qualification(tmp_path, monkeypatch, split):
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('test_improvement_launcher',scripts/'run_improvement_heat.py')
    launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
    # A conflicting module already loaded by the orchestrator must not leak
    # into the subprocess that constructs a particular frozen world.
    monkeypatch.setitem(sys.modules,'race_lidar.hotlap_driver',SimpleNamespace(RacingReference=None))
    source=tmp_path/'frozen';package=source/'src/race_lidar';package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'hotlap_driver.py').write_text(
        'class RacingReference:\n'
        '    @classmethod\n'
        '    def from_npz(cls,path):\n'
        '        value=cls();value.length=731.5;return value\n')
    (source/'scripts').mkdir()
    (source/'scripts/generate_tournament_world.py').write_text(
        'import argparse,json\nfrom pathlib import Path\n'
        'from race_lidar.hotlap_driver import RacingReference\n'
        'p=argparse.ArgumentParser();p.add_argument("--config");p.add_argument("--output");a=p.parse_args()\n'
        'Path(a.output).write_text("frozen:"+str(RacingReference.from_npz(None).length))\n')
    for name in ('tournament_director','tournament_racer'):
        (source/'webots/controllers'/name).mkdir(parents=True)
    (source/'manifest.json').write_text(json.dumps({'snapshot_sha256':'test-frozen',
        'metadata':{'policies':[{'policy':'reference','label':'Reference'}],
                    'policy_parameters':{'visible_distance_m':70.}}}))
    args=argparse.Namespace(output=tmp_path/'run',source_root=source,reference=source/'reference.npz',
        visibility=None,policies=None,single_policy='reference',seed=79,cars=1,spacing=12.,rotation=0,
        grid_jitter=0.,scenario='parking',ego_lane=0.,ego_progress=0.,obstacle_progress=100.,
        blocked_side='right',secondary_cars=0,secondary_url='http://unused',no_video=True,
        capacity=True,seconds=80.,pointcloud_stride=2,vision_url='http://unused',detector_sha='test',
        split=split)
    config_path,world=launcher.prepare(args)
    config=json.loads(config_path.read_text())
    assert config['reference_length_m']==731.5
    assert config['entrants'][0]['start_progress_m']==601.5
    assert config['policy_parameters']=={'visible_distance_m':70.}
    assert config['excluded_from_training'] is True
    assert world.read_text()=='frozen:731.5'
