# Attribution

First-party project code is covered by the existing repository's [MIT license](LICENSE).
Implementation, experiment automation, analysis and documentation used AI coding
assistance; see [CONTRIBUTIONS.md](CONTRIBUTIONS.md).

VoxelNeXt is external research software. Its repository, pinned source commit,
configuration and original checkpoint hashes are recorded in
[artifacts/downloads.json](artifacts/downloads.json). Acquire it upstream and
preserve its license and attribution. No VoxelNeXt source tree or detector
weights are redistributed here.

Webots vehicle PROTOs, textures and sounds are not bundled. Their pinned
[source manifest](webots/protos/vehicle/sources.json) and
`scripts/fetch_vehicle_assets.py` retain their upstream identity. Downloaded
assets remain subject to their own licenses, including the Webots asset license
identified in the original PROTO headers. The project MIT license does not
replace those terms.

The approximate course reference comes from public OpenF1 telemetry. Acquisition
and limitations are documented in [docs/f1_reference.md](docs/f1_reference.md).
It is not a surveyed circuit or an exact Formula 1 racing line. Full reference
data, gated A2RL data and proprietary F1 feeds are not bundled.

The small measured cloud, cached predictions/labels and demo are outputs from
this project's simulated scenes. The caches support offline evaluation; they
never supply ground truth to a detector or driving policy. The recorded demo
contains Webots-rendered assets and is research footage, not a grant of rights
to third-party source assets or trademarks.
