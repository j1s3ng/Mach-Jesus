# Source provenance

This tree was prepared from the verified CMPE249 export at commit
`ca985876a6539cf4630091d72554a91501ae3eef` (November 14, 2026).

The JSON files in this directory are historical receipts for that export. They
preserve its original file paths, hashes, test results and campaign tools.
They are not a current manifest of this repository: documentation has changed
and these receipts were moved from the export root. Their original source and
evidence paths identify archived inputs, not public download locations.

During the initial export preparation, runtime, regression-test and
cached-reproduction files were copied unchanged. That preparation changed only
the validation/import order in
`scripts/train_race_driver.py`: embedded metadata and original source paths are
checked before optional PyTorch imports and before feature arrays load. This
fix does not alter historical experiment files, training settings or fit logic.
`PORTABLE_TESTS.json` remains at the repository root because it
controls the default test collection. Historical campaign tests require
separate archived inputs and explicit opt-in.

The initial November 18 maintenance changed nine files: the hotlap and engineering
launchers (`scripts/run_hotlap.py`, `scripts/run_improvement_heat.py`), packaging
(`pyproject.toml`), portable test collection (`PORTABLE_TESTS.json`,
`tests/conftest.py`), regression/optional-dependency checks
(`tests/test_launchers.py`, `tests/test_test_collection.py`,
`tests/test_learned_driver.py`), and CPU CI (`.github/workflows/ci.yml`).
The [local verification record](../verification.md) describes these fixes and
their checks. Actor runtime modules, source overlays, frozen evidence and
cached-reproduction inputs remain unchanged by this maintenance.

The later [motion reproduction](../../reproduction/control-cadence/README.md)
adds three saved solo recordings and their CPU metric checks. Its tests,
dependency lock, CI command and navigation are new. The ignore rules explicitly
include its three small NPZ files. Existing cached detector inputs and metric
implementations retain their original bytes; the reproduction README is now a
current recipe catalog.

The [documentation amendment ledger](DOCUMENTATION_AMENDMENTS.json) separately
records archive-link/navigation changes to three experiment READMEs. Exact
manifest-pinned originals are retained as text under `history/`; their embedded
media paths describe the historical snapshot. Frozen manifests were preserved.

The [study record](../final_artifact_provenance.json) preserves the completed
experiment and artifact identities. Its publication/deferred fields describe
the November 14 export. Model/data downloads remain unavailable; a GitHub source
repository alone does not supply the full private evidence or a complete race
reproduction.

| Included artifact | SHA-256 |
| --- | --- |
| [Original November 14 report](https://github.com/j1s3ng/lidar-racing/blob/6ff9ef70dde7a50282f43bf612ac67a48df5ea26/docs/report.pdf) | `de0768ebf081319d252da4bad7e222d2b03c941f72270c9ca1bf8f7c22165d84` |
| [116-second demo](../demo.mp4) | `9c9269968cc1af98287c8a645f6dede2ff5a02ec5e110369cb6c4ddc0fd3df6d` |

The original measured source archives and immutable delivery remain separate
from this public-facing working copy.

The November 19 [project-title revision](project-rename.json) records current
asset hashes and their original counterparts. It changes presentation titles,
repository/media links and filenames, with unchanged experimental data, model
weights, runtime source and numerical results. Original artifacts remain in the
pre-revision Git commit and verified local backups.

The [documentation clarity amendment](CLARITY_AMENDMENT.json) records later
reader-facing corrections, including current video, repository and CI status.
Earlier receipts retain their original snapshot hashes; scientific data and
runtime source are unchanged.
