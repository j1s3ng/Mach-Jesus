#!/usr/bin/env python3
"""Build the closed four-arm DEVELOPMENT table and vector figures; CPU only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "runs/improvement-12h/training"
DIAG = ROOT / "runs/improvement-12h/diagnostics"
AUDITS = {
    "head-lr1e-4-s7": DIAG / "controlled-head-parameter-audit-001.json",
    "full-uniform-lr1e-4-s7": DIAG / "controlled-full-uniform-high-parameter-audit-001.json",
    "full-uniform-lr1e-5-s7": DIAG / "controlled-full-uniform-low-parameter-audit-racer001.json",
    "full-mixed-head1e-4-backbone1e-5-s7": DIAG / "controlled-full-mixed-parameter-audit-racer001.json",
}
LABELS = {"head-lr1e-4-s7": "Head\n1e−4", "full-uniform-lr1e-4-s7": "Full uniform\n1e−4",
          "full-uniform-lr1e-5-s7": "Full uniform\n1e−5", "full-mixed-head1e-4-backbone1e-5-s7": "Full mixed\n1e−4 / 1e−5"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(analysis_path):
    analysis = json.loads(analysis_path.read_text())
    if not analysis["all_four_completed_and_verified"] or not analysis["matched_sampling_comparison_valid"]:
        raise ValueError("A final matched figure requires all four closed, verified, sampling-matched attempts; preserve partial/failure analysis separately")
    params = {}
    for arm in analysis["attempts"]:
        path = AUDITS[arm["arm_id"]]
        audit = json.loads(path.read_text())
        for filename, digest in audit["source_sha256"].items():
            if sha(filename) != digest:
                raise ValueError("Parameter audit input/source changed:" + filename)
        run = Path(arm["output"])
        checkpoint = run / "candidate_step_1200.pth"
        if audit["source_sha256"].get(str(checkpoint.resolve())) != arm["candidate_checkpoint_sha256"]:
            raise ValueError("Actual parameter audit belongs to another checkpoint")
        if audit["mode"] != arm["mode"] or audit["training_steps"] != 1200:
            raise ValueError("Parameter audit mode/budget differs")
        if not audit["all_BN_running_buffers_unchanged"] or audit["BN_running_buffer_count"] != 204:
            raise ValueError("Normalization freezing proof differs")
        changed = audit["groups"]["backbone"]["changed_parameter_tensors"]
        if changed != (0 if arm["mode"] == "head" else 121):
            raise ValueError("Actual backbone updates do not match declared scope")
        params[arm["arm_id"]] = {"audit_path": str(path), "audit_sha256": sha(path), **audit}
    common = analysis["common_gt_across_verified_available_arms"]
    if len(common["models"]) != 5 or common["matches"] <= 0:
        raise ValueError("The original and all four models need a common GT cohort")
    rows = []
    for arm in analysis["attempts"]:
        metrics = arm["development_score_0p35"]
        audit = params[arm["arm_id"]]
        rows.append({"arm_id": arm["arm_id"], "label": LABELS[arm["arm_id"]],
            "head_lr": arm["head_lr"], "backbone_lr": arm["backbone_lr"],
            "trainable_scalar_parameters": audit["trainable_scalar_parameters"],
            "trainable_tensors": arm["trainable_parameter_tensors"],
            "actual_changed_head_tensors": audit["groups"]["head"]["changed_parameter_tensors"],
            "actual_changed_backbone_tensors": audit["groups"]["backbone"]["changed_parameter_tensors"],
            "BN_running_buffers_unchanged": 204, "precision": metrics["precision"], "recall": metrics["recall"],
            "false_positives": metrics["false_positives"], "false_positives_per_frame": metrics["false_positives_per_frame"],
            "far_recall_30_50m": metrics["far_30_50m"]["recall"],
            "common_all5_xy_center_mae_m": common["xy_center_mae_m"][arm["arm_id"]],
            "status_wall_seconds": arm["wall_seconds_from_status_timestamps"],
            "phase_timer_includes_fit_and_adapted_eval_seconds": arm["timer_training_seconds_includes_adapted_eval"],
            "candidate_checkpoint_sha256": arm["candidate_checkpoint_sha256"], "evidence_verified": True})
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One matched seed7 comparison on the frozen6144-frame development run, not qualification, officialAP or a racing outcome.",
        "analysis_sha256": sha(analysis_path), "analysis_source_sha256": sha(ROOT / "scripts/compare_controlled_ablation.py"),
        "report_source_sha256": sha(__file__), "plan_sha256": analysis["plan_sha256"],
        "models_have_same_total_inference_parameters": 7981398, "scalar_counts_exclude_buffers": True,
        "fixed_score_cutoff": .35, "development_frames": 6144, "targets": 8155,
        "common_all5_gt_targets": common["matches"], "original": analysis["original_pretrained_at_0p35"],
        "original_common_xy_center_mae_m": common["xy_center_mae_m"]["original_pretrained"],
        "rows": rows, "parameter_audits": params,
        "matched_controls": "B2/1200steps/seed7,600positive600empty,2400draws1398unique, identical TRAIN-only miner weights and sample frame order, original initialization, frozenBN, source/data/config identities.",
        "failure_accounting": "All four predeclared actual attempts are present and verified. Unsuccessful and partial states remain in the earlier frozen analyzer artifacts; none replaced by another seed.",
        "timing_limit": "Status wall includes loading/mining/fitting/6144 adapted evaluation with baseline predictions cached. The phase timer also includes adapted evaluation. Neither is pure optimizer throughput or an isolated GPU benchmark.",
        "statistical_limit": "One seed and one correlated physical development run do not establish statistically general optimizer superiority.",
        "selection_performed_by_report": False}


def save_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    rows = report["rows"]
    labels = ["Original\npretrained"] + [row["label"] for row in rows]
    colors = ["#7c8794", "#c76151", "#24807a", "#477bb4", "#a17cb2"]
    old = report["original"]
    data = [("Recall at score .35", "%", [100*old["recall"]] + [100*r["recall"] for r in rows]),
            ("False positives on 6,144 frames", "count", [old["false_positives"]] + [r["false_positives"] for r in rows]),
            ("Far-target recall (30–50 m)", "%", [100*old["far_30_50m"]["recall"]] + [100*r["far_recall_30_50m"] for r in rows]),
            (f"Center error on the same {report['common_all5_gt_targets']:,} GT matches", "m", [report["original_common_xy_center_mae_m"]] + [r["common_all5_xy_center_mae_m"] for r in rows])]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.), constrained_layout=True)
    for axis, (title, unit, values) in zip(axes.flat, data):
        bars = axis.bar(range(5), values, color=colors)
        axis.set_xticks(range(5), labels)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        if unit == "%": axis.set_ylim(0, 108)
        else: axis.set_ylim(0, max(values)*1.2)
        axis.bar_label(bars, labels=[f"{v:.1f}" if unit=="%" else f"{v:.3f}" if unit=="m" else f"{v:.0f}" for v in values], padding=3)
    fig.suptitle("Matched source, samples and 1,200-step budget — seed 7\nDevelopment comparison; custom 2 m XY-center matching", fontsize=13)
    for suffix in ("pdf", "svg", "png"): fig.savefig(output/("controlled_metrics."+suffix), dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12., 3.9), constrained_layout=True)
    labels = [r["label"] for r in rows]
    specifications = [("Trainable parameters", "millions", [r["trainable_scalar_parameters"]/1e6 for r in rows]),
                      ("Backbone tensors actually changed", "out of 121", [r["actual_changed_backbone_tensors"] for r in rows]),
                      ("Observed total status wall time", "minutes", [r["status_wall_seconds"]/60 for r in rows])]
    for axis,(title,unit,values) in zip(axes,specifications):
        bars=axis.bar(range(4),values,color=colors[1:]);axis.set_xticks(range(4),labels);axis.set_title(title);axis.set_ylabel(unit)
        axis.grid(axis="y",alpha=.2);axis.set_axisbelow(True);axis.set_ylim(0,max(values)*1.22)
        axis.bar_label(bars,labels=[f"{v:.3f}" if unit=="millions" else f"{v:.0f}" if unit=="out of 121" else f"{v:.2f}" for v in values],padding=3)
    fig.suptitle("Same 7.981M-parameter inference architecture in every arm\nAll 204 BN running buffers preserved; timing includes mining, fit and development inference",fontsize=12)
    for suffix in ("pdf","svg","png"):fig.savefig(output/("controlled_training_scope."+suffix),dpi=180)
    plt.close(fig)


def write_table(report, output):
    lines=["# Controlled perception comparison", "", "All four fixed seed-7 attempts completed and passed identity, checkpoint, gradient, mining and sample-order audits. Results are development diagnostics at score 0.35, not qualification or racing results.", "",
        "| Arm | Head / backbone LR | Trainable parameters | BB tensors changed | Recall | FP / 6,144 frames | Far recall | Common-center MAE | Wall |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    old=report["original"]
    lines.append(f"| Original pretrained | — | — | — | {old['recall']:.2%} | {old['false_positives']} | {old['far_30_50m']['recall']:.2%} | {report['original_common_xy_center_mae_m']:.3f} m | — |")
    for row in report["rows"]:
        lr=f"{row['head_lr']:g} / "+("frozen" if row['backbone_lr']==0 else f"{row['backbone_lr']:g}")
        lines.append(f"| {row['label'].replace(chr(10),' ')} | {lr} | {row['trainable_scalar_parameters']:,} | {row['actual_changed_backbone_tensors']}/121 | {row['recall']:.2%} | {row['false_positives']} | {row['far_recall_30_50m']:.2%} | {row['common_all5_xy_center_mae_m']:.3f} m | {row['status_wall_seconds']/60:.2f} min |")
    lines += ["",f"Center errors use the identical {report['common_all5_gt_targets']:,} GT IDs matched by all five models, including the original. Far recall uses 30–50 m targets. All 204 batch-normalization running buffers were unchanged. Full fits changed all 121 backbone parameter tensors; the head-only fit changed none. Every model retains the same 7,981,398 scalar inference parameters; the table counts parameters enabled for training.","",
        "Every arm used the same 1,200 updates, batch size 2, 600 positive/600 empty batches, 2,400 frame draws (1,398 unique), TRAIN-only miner, frozen initialization/configuration and source. The old mixed-LR pilot and optional seed 17 are excluded from this matched table.","",
        report["timing_limit"],"",report["statistical_limit"],"","No new model, score threshold or deployment was selected by this report. The separate augmentation arm is outside these four-arm results."]
    (output/"README.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("Preserve previous summaries")
    report=assemble(args.analysis.resolve());args.output.mkdir(parents=True)
    (args.output/"controlled_table.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    write_table(report,args.output);save_figures(report,args.output)
    print(json.dumps({"output":str(args.output),"arms":len(report["rows"]),"common_targets":report["common_all5_gt_targets"]}))


if __name__=="__main__":main()
