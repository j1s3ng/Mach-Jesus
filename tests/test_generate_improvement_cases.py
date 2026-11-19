import argparse
import ast
import copy
from pathlib import Path

import pytest

from scripts.generate_improvement_cases import formal_cases, verify_case
from scripts.improvement_ledger import OFFICIAL_DETECTOR_SHA256, digest, scenario_identity


def test_formal_cases_cover_distinct_seed_spacing_starts_and_all_three_hazards():
    holdout,confirmation = formal_cases("holdout"),formal_cases("confirmation")
    assert len(holdout)==15
    assert {c["scenario"]["seed"] for c in holdout}.isdisjoint({c["scenario"]["seed"] for c in confirmation})
    assert all(c["excluded_from_training"] for c in holdout+confirmation)
    for count in (4,8,20):
        cases=[c for c in holdout if c["kind"]=="race" and c["cars"]==count]
        assert len(cases)==4
        assert {c["scenario"]["spacing"] for c in cases}=={12.,14.,16.,18.}
        assert len({c["scenario"]["seed"] for c in cases})==4
        assert all(c["scenario"]["grid_jitter"]==.3 for c in cases)
    hazards=[c["scenario"] for c in holdout if c["is_hazard"]]
    assert [(h["obstacle_progress"],h["ego_progress"],h["blocked_side"],h["cars"]) for h in hazards]==[
        (300.,0.,"right",2),(1000.,900.,"right",3),(3000.,2900.,"left",3)]


def test_generated_argv_matches_actual_launcher_parser_and_scenario_hash_expression(tmp_path):
    # Execute only argparse declarations and the identity expression from the
    # actual launcher, never prepare(), process creation, or simulator code.
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/"scripts/run_improvement_heat.py").read_text())
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="main")
    parser=argparse.ArgumentParser()
    declarations=[node for node in main.body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
                  and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name)
                  and node.value.func.value.id=="p" and node.value.func.attr=="add_argument"]
    for declaration in declarations:
        exec(compile(ast.Module(body=[declaration],type_ignores=[]),"launcher_argparse","exec"),
             {"p":parser,"Path":Path,"ROOT":root,"OFFICIAL_DETECTOR_SHA256":OFFICIAL_DETECTOR_SHA256})
    assignment=next(node for node in main.body if isinstance(node,ast.Assign)
                    and any(isinstance(target,ast.Name) and target.id=="scenario" for target in node.targets))
    for case in formal_cases("holdout")+formal_cases("confirmation"):
        args=parser.parse_args(["--output",str(tmp_path/"unused"),*case["cli_args"]])
        actual=eval(compile(ast.Expression(assignment.value),"launcher_scenario","eval"),
                    {"a":args,"scenario_identity":scenario_identity})
        assert digest(actual)==case["scenario_sha256"]
        assert verify_case(case)==actual
        assert "--single-policy" not in case["cli_args"]
        assert "--visibility" not in case["cli_args"]


def test_case_identity_rejects_changed_challenge_or_hidden_policy_choice():
    case=formal_cases("holdout")[0]
    changed=copy.deepcopy(case);changed["scenario"]["spacing"]+=1
    with pytest.raises(ValueError,match="identity"):
        verify_case(changed)
    changed=copy.deepcopy(case);changed["scenario"]["single_policy"]="easier-driver"
    with pytest.raises(ValueError,match="treatment"):
        verify_case(changed)
    assert scenario_identity({**case["scenario"],"single_policy":"new","visibility":70.})==case["scenario"]


def test_engineering_capacity_identity_allows64_without_expanding_formal_strata():
    assert scenario_identity({"cars":64})["cars"]==64
    with pytest.raises(ValueError,match="scenario size"):
        scenario_identity({"cars":65})
    assert {case["cars"] for case in formal_cases("holdout") if not case["is_hazard"]}=={4,8,20}
