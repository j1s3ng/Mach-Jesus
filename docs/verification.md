# Local verification

The cleaned checkout was verified on macOS arm64 with Python 3.11.6 in a fresh isolated CPU environment. No learned detector or vehicle simulation is needed for these checks.

| Check | Result |
| --- | --- |
| Portable regression suite | **284 passed, 7 optional skips**; no failures. |
| Cached qualification reproduction | 8,955 frames reproduced within archived tolerances. |
| Cached native-view reproduction | 6,144 GT frames and 36,864 prediction rows reproduced. |
| Cached rotation reproduction | 512 GT frames, 3,072 prediction rows and 24 reports reproduced. |
| Saved motion reproduction | All 27,801 native samples across three solo recordings reproduced; all frozen metric fields matched. |
| Synthetic CPU demo | 20 frames generated, replayed and scored. |
| Initial editable install | Imports, CLI help and `doctor` passed outside the source checkout. |
| Built wheel | 30 Python modules from exactly `race_lidar` and `training`; every module byte matched source. |
| Installed wheel | Imports, installed CLI help and `doctor` passed outside the checkout without Torch. |

Optional skips reflect unavailable optional dependencies in that CPU environment. The NumPy driver and core/control/transport tests still run. The synthetic clustering demo is a toy baseline; its low recall is not the neural racing result.

The final CPU environment used NumPy 1.26.4, SciPy 1.16.2 and Shapely 2.1.2.
The earlier 265-test cleanup check was expanded with 19 saved-motion tests.
The [machine-readable record](verification.json) binds the tested source and
wheel hashes. The three small motion NPZ files are explicitly included by the
ignore rules, and the reproduction extra supplies the required dependency lock.

## Fixes covered by regression tests

- The default zero-backmarker hotlap now generates its world on a clean checkout; explicit custom worlds remain supported.
- The first engineering run creates its simulator-lock directory, releases its lock, and rejects nonfinite time/limit inputs before writing experiment outputs.
- Installable distributions include the training/inference helper package required by the CLI's neural backend. No weights or vendor frameworks are bundled.
- Nine previously omitted self-contained test files now run by default. New unlisted tests also run; only explicitly listed archived-campaign tests require opt-in. Their data requirements remain documented in [setup](setup.md).

The [CPU CI workflow](../.github/workflows/ci.yml) checks installation, portable tests, saved-data recipes and the synthetic demo on Ubuntu 24.04 / Python 3.11. It has read-only repository permissions. Hosted results are available in the repository's Actions tab; the original cleanup record above remains historical.

These checks verify the public CPU paths and packaging. Live CUDA inference and Webots physics require the external artifacts described in [setup](setup.md). The [frozen study evidence](final_evidence.md) and [latest engineering measurements](../experiments/apex-tracking/README.md) are separate records of those experiments.
