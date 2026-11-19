#!/usr/bin/env python3
"""Assemble explicit new split roles using immutable exported-cloud hard links."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(training_sources, development_source, protocol_path, output):
    if output.exists(): raise ValueError("Use a fresh output directory")
    protocol = json.loads(protocol_path.read_text())
    metadata, source_rows, sources, source_hashes = None, {"train": [], "val": []}, [], {}
    for destination, inputs in (("train", training_sources), ("val", [development_source])):
        for directory, split in inputs:
            current = json.loads((directory/"dataset.json").read_text())
            if metadata is None: metadata = copy.deepcopy(current)
            for filename, digest in current.get("source_file_sha256", {}).items():
                if filename in source_hashes and source_hashes[filename] != digest:
                    raise ValueError("Source provenance hashes conflict")
                source_hashes[filename] = digest
            for key in ("min_measured_points_per_box", "evaluation_radius_m", "point_in_box_tolerance_m", "ground_origin", "timestamp_join"):
                if current[key] != metadata[key]: raise ValueError("Source exported label/geometry contracts differ")
            manifest = directory/(split+".jsonl")
            if sha(manifest) != current["manifest_sha256"][split]: raise ValueError("Source manifest changed")
            values = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
            source_rows[destination].extend((row, directory) for row in values)
            sources.append({"dataset": directory.name, "source_split": split, "new_split": destination,
                            "source_dataset_sha256": sha(directory/"dataset.json"), "source_manifest_sha256": sha(manifest), "frames": len(values)})
    train_groups = {row["scene_id"] for row, _ in source_rows["train"]}
    val_groups = {row["scene_id"] for row, _ in source_rows["val"]}
    if train_groups != set(protocol["training_groups"]) or val_groups != set(protocol["development_groups"]) or train_groups&val_groups:
        raise ValueError("Physical scene roles differ from the declared iteration protocol")
    train_hashes = {row["points_sha256"] for row, _ in source_rows["train"]}
    removed = [{"scene_id": row["scene_id"], "frame_id": row["frame_id"], "points_sha256": row["points_sha256"], "reason": "exact_training_cloud_hash"}
               for row, _ in source_rows["val"] if row["points_sha256"] in train_hashes]
    source_rows["val"] = [(row, directory) for row, directory in source_rows["val"] if row["points_sha256"] not in train_hashes]
    assert not train_hashes&{row["points_sha256"] for row, _ in source_rows["val"]}
    output.mkdir(parents=True);(output/"points").mkdir()
    records = {}
    for split in ("train", "val"):
        seen, records[split] = set(), []
        for source, directory in source_rows[split]:
            row = copy.deepcopy(source);key = (row["scene_id"], row["frame_id"])
            if key in seen: raise ValueError("A source frame was included twice")
            seen.add(key)
            point = (directory/row["points_file"]).resolve()
            if not point.is_relative_to(directory.resolve()) or sha(point) != row["points_sha256"]:
                raise ValueError("Source point file changed or escapes its export")
            row["points_file"] = "points/"+row["frame_id"]+".npy"
            target = output/row["points_file"]
            if target.exists():
                if sha(target) != row["points_sha256"]: raise ValueError("Point filename collision")
            else: os.link(point, target)
            records[split].append(row)
        (output/(split+".jsonl")).write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in records[split]))
    metadata.update(splits={split: {"frames": len(values), "positive_frames": sum(bool(row["gt_boxes"]) for row in values),
                                   "visible_car_boxes": sum(len(row["gt_boxes"]) for row in values), "groups": sorted({row["scene_id"] for row in values})}
                            for split, values in records.items()},
                    manifest_sha256={split: sha(output/(split+".jsonl")) for split in records},
                    source_exports=sources, assembly_protocol=protocol, assembly_protocol_sha256=sha(protocol_path),
                    source_file_sha256=source_hashes,
                    point_storage="hard links to immutable existing measured exports; never mutate either link",
                    development_role="development only, no gradient/mining; not fresh qualification",
                    excluded_duplicate_validation_frames=removed,
                    qualification_subset_scope="run-separated development with deterministic input-only training-cloud deduplication",
                    input_frame_counts_by_group={"val": {group: sum(row["scene_id"] == group for row, _ in source_rows["val"])+sum(row["scene_id"] == group for row in removed) for group in val_groups}})
    (output/"dataset.json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def source(value):
    directory, split = value.rsplit(":", 1)
    if split not in ("train", "val"): raise argparse.ArgumentTypeError("Expected DIRECTORY:train or DIRECTORY:val")
    return Path(directory), split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-sources", type=source, nargs="+", required=True)
    parser.add_argument("--development-source", type=source, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = assemble(args.training_sources, args.development_source, args.protocol, args.output)
    print(json.dumps({key: metadata[key] for key in ("splits", "manifest_sha256", "excluded_duplicate_validation_frames")}, indent=2))


if __name__ == "__main__":
    main()
