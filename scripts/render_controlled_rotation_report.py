#!/usr/bin/env python3
"""Render vector figures and a short report from closed CPU-scored diagnostics."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = {"original": "Original (cached)", "head-lr1e-4-s7": "Head 1e-4",
    "full-uniform-lr1e-4-s7": "Full 1e-4", "full-uniform-lr1e-5-s7": "Full 1e-5",
    "full-mixed-head1e-4-backbone1e-5-s7": "Mixed LR", "full-mixed-rotaug4-s7": "Mixed LR + rotations"}
COLORS = {name: color for name, color in zip(LABELS, ("#555555", "#2465a5", "#ba4248", "#be801d", "#7956a3", "#168579"))}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def render(analysis_path, unrotated_path, output):
    output = Path(output)
    if output.exists(): raise ValueError("Preserve earlier figures; choose a new report directory")
    report, unrotated = read(analysis_path), read(unrotated_path)
    audit = report["independent_source_inventory_resource_audit"]
    if not audit["verified"]: raise ValueError("An independent source/inventory/resource audit is required")
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    lookup = {(row["model"], row["degrees"]): row for row in report["reports"]}
    specs = (("recall", "Visible target recall", "%"), ("directed_heading_rad", "Directed heading error", "degrees"),
        ("undirected_box_heading_rad", "Undirected box heading error", "degrees"), ("bev_iou", "BEV IoU", "IoU"),
        ("upright_3d_iou", "Upright 3D IoU", "IoU"), ("height_abs_error_m", "Absolute height error", "m"))
    def figure(models, cohort, count, name, title):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
        for axis, (metric, label, units) in zip(axes.flat, specs):
            for model in models:
                values = []
                for angle in report["angles"]:
                    row = lookup[(model, angle)]
                    if metric == "recall": value = row["recall_on_scored_inputs"] * 100
                    else:
                        paired = row.get(cohort); geometry = paired.get("geometry") if paired else None
                        value = geometry[metric]["mean"] if geometry else np.nan
                        if units == "degrees": value = np.degrees(value)
                    values.append(value)
                axis.plot(range(4), values, marker="o", linewidth=1.7, color=COLORS[model], label=LABELS[model],
                          linestyle="--" if model == "original" else "-")
            axis.set_xticks(range(4), ["−90°", "0°", "+90°", "180°"]); axis.set_title(label)
            axis.set_xlabel("Rigid measured-cloud and GT rotation"); axis.set_ylabel(units); axis.grid(alpha=.23)
            if metric == "recall": axis.set_ylim(0, 103)
            if units == "IoU": axis.set_ylim(0, 1.02)
        axes.flat[0].legend(frameon=False, fontsize=8)
        fig.suptitle(title + f"\nGeometry: same {count} target instances across models/angles; recall: all174 targets per angle", fontsize=11)
        for suffix in ("pdf", "svg", "png"): fig.savefig(output / (name + "." + suffix), dpi=170)
        plt.close(fig)
    figure(["original", *report["core_models"]], "fixed_core4_plus_original_all_angles_cohort",
        report["core4_plus_original_fixed_all_angles_common_targets"], "core4_rotation",
        "Four controlled seed7 fits on128 known development frames (RAW .425)")
    mixed, augmented = "full-mixed-head1e-4-backbone1e-5-s7", "full-mixed-rotaug4-s7"
    figure(["original", mixed, augmented], "separate_optional_plus_core_and_original_cohort",
        report["optional_extended_common_targets"], "augmentation_rotation",
        "Predeclared TRAIN-rotation intervention: development stress diagnostic")
    lines = ["# Controlled detector rotation diagnostic", "",
        "All five fitted checkpoints completed the same128 known development frames at four rigid rotations, with RAW confidence0.425. "
        "The independently audited standalone run recorded2,560 successful scored forwards plus10 warmups, no retries/errors, "
        "and68.619 seconds of executor wall time. Owned GPU/CPU cleanup and three service checks preceded queue resumption.", "",
        "TRAIN rotation improved the mixed model's rotated footprint estimates, with a tradeoff in directed front/back orientation. "
        "On the same41 targets,180° directed-heading error fell2.853→1.032rad and BEV IoU rose0.730→0.828; "
        "at0°, directed-heading error rose0.105→0.564rad. These are diagnostics on correlated, already inspected measurements.", "",
        "| Checkpoint | Recall−90/0/+90/180 (%) | FP/512 | Hostpredict p95(ms) | PyTorch allocated peak(GiB) |",
        "|---|---|---:|---:|---:|"]
    memory = {row["arm_id"]: row.get("per_arm_peak_torch_allocated_bytes") for row in audit["per_arm_memory_reports"]}
    for model in report["all_models"]:
        recalls = "/".join(f"{100*lookup[(model,a)]['recall_on_scored_inputs']:.1f}" for a in report["angles"])
        fp = sum(lookup[(model,a)]["false_positives"] for a in report["angles"])
        latency = audit["by_model"][model]["predict_host_wall_p95_ms"]
        peak = memory.get(model); peak_text = f"{peak/2**30:.3f}" if peak is not None else "—"
        lines.append(f"|{LABELS[model]}|{recalls}|{fp}|{latency:.2f}|{peak_text}|")
    lines += ["", "Recall uses174 visible targets per angle. The fixed core-only intersection contains102 target instances; "
        "adding the previously measured original baseline gives41. Including augmentation also gives41. "
        "Per-angle and fixed-intersection geometry counts remain separate in result.json; cached baseline requires no new inference.", "",
        "The already captured unrotated6,144-frame DEV caches show a separate tradeoff:", "",
        "| RAW cutoff | Mixed recall/FP | Rotated-TRAIN recall/FP | Same-GT matches |", "|---:|---:|---:|---:|"]
    for point in unrotated["operating_points"]:
        a, b = point["models"]["mixed_no_rotaug"], point["models"]["mixed_train_rotaug"]
        lines.append(f"|{point['raw_score_threshold']}|{100*a['recall']:.2f}%/{a['false_positives']}|"
                     f"{100*b['recall']:.2f}%/{b['false_positives']}|{point['same_GT_common_matches']}|")
    lines += ["", "The RAW0.35 counts exactly reproduce saved training-evaluation metrics; cached effective floor0.1 supports both cutoffs. "
        "No raw cloud, training, threshold fitting, deployment or qualification was performed for this cache comparison.", "",
        "Hostpredict wall includes CPU preparation, transfer, model and NMS. Memory is the reset per-model PyTorch allocated peak "
        "(torch.cuda.max_memory_allocated), excluding CUDA context, custom spconv and other non-PyTorch allocations. It is not total "
        "process/device memory or an NVML peak. Two resident detector services stayed loaded. Neither timing nor memory is an isolated hardware benchmark. "
        "All models share the architecture; source, checkpoint, tensor, input and actual PRO-process identities are bound in result.json.", "",
        "**Scope:** known DEVELOPMENT sensitivity and a single matched training seed. No independent generalization, causal-shortcut proof, "
        "official A2RL/nuScenes AP, model/racer selection or fresh qualification claim. Original pilot observations remain historical; "
        "only the new controlled mixed checkpoint is the augmentation control.", "",
        "Vector figures: core4_rotation.pdf/.svg and augmentation_rotation.pdf/.svg. Full source/inventory/cleanup audits and "
        "all per-angle counts are in ../result.json; exact hashes accompany this report in manifest.json.", ""]
    (output / "README.md").write_text("\n".join(lines))
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "analysis_sha256": sha(analysis_path),
        "unrotated_diagnostic_sha256": sha(unrotated_path), "writer_source_sha256": sha(__file__),
        "artifact_sha256": {path.name: sha(path) for path in output.iterdir() if path.is_file()},
        "SVG_raster_images": {path.name: path.read_text().count("<image") for path in output.glob("*.svg")},
        "new_inference_or_training": False, "qualification_or_selection": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True); parser.add_argument("--unrotated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    result = render(args.analysis, args.unrotated, args.output)
    print(json.dumps({"output": str(args.output), "SVG_raster_images": result["SVG_raster_images"]}))
