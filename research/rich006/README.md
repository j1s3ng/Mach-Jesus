# Rich006 archived training source

These four files are exact copies of the actual fitted model's frozen teacher,
fitter, and orchestration sources. Their SHA256 values are in source_manifest.json.
The portable 48-input runtime remains in src/race_lidar/rich_driver.py.

The archive preserves relative paths from the original campaign. Do not invoke
the archived CLI main as a clean-clone quickstart: it rebuilds inputs from
unbundled, authorized engineering runs and depends on its original archive layout.
These filenames preserve attribution and inspection, not relocated provenance.

Actual numeric fitting used the immutable fitting_examples.npz companion and
Rich005 seed 17 warm weights, with seed 17, 80 epochs, batch 1024, learning rate 0.0003,
and four CPU threads. The selected epoch was 1. Its final exported checkpoint SHA
is listed as `rich006_seed17` in artifacts/downloads.json. A future portable fit
wrapper must explicitly bind these exact inputs and preserve the frozen fitter;
it would generate a new artifact with its own local metadata. No fitting is run
when this repository is installed or when the cached metric commands execute.

The teacher and full raw source reports are separate: private training reports
and observation logs are not copied here. Raw source provenance and processed
fitting arrays remain separate companion inputs; qualification/final inputs
must never be used to recreate training rows.
