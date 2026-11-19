# Local archive

The complete checkout was backed up before cleanup, including Git metadata,
untracked files and ignored working output. Superseded media and generated
working files were then moved byte-for-byte to the local project archive
`project-cleanup-20261118-001`. Nothing in this catalog is a hosted download.

The latest demo, native recording, replay and poster remain in `docs/`, as do
the final report/presentation, original study demo and the presentation's
matching chase recording. Source overlays, cached reproduction inputs, figures
and scientific evidence remain in the checkout.

## Historical media

Links from older notes now lead to this catalog rather than missing videos.
Original JSON receipts retain their historical paths/hashes; those fields are
identities of the recorded artifact, not current availability assertions.
[archive_manifest.json](archive_manifest.json) records current archive status,
original paths, sizes and SHA-256 values. Existing frozen JSON/source receipts
were not rewritten during cleanup.

Three experiment READMEs now use archive links and explain their frozen
inventory scope. Their historical manifest hashes therefore bind the original
README bytes, not the edited navigation copies. The
[documentation amendment ledger](provenance/DOCUMENTATION_AMENDMENTS.json)
binds each unchanged manifest, exact retained original text and current README.

| Archived file | Size | Original role |
| --- | ---: | --- |
| <a id="apex-demo-mp4"></a>`Apex_Demo.mp4` | 11.32 MiB | Earlier apex/gap presentation |
| <a id="apex-normal-speed-mp4"></a>`Apex_Normal_Speed.mp4` | 14.92 MiB | Earlier apex/gap native recording |
| <a id="middle-pass-demo-mp4"></a>`Middle_Pass_Demo.mp4` | 0.72 MiB | Controlled middle-pass presentation |
| <a id="middle-pass-normal-speed-mp4"></a>`Middle_Pass_Normal_Speed.mp4` | 0.60 MiB | Controlled middle-pass native recording |
| <a id="pass-first-demo-mp4"></a>`Pass_First_Demo.mp4` | 11.86 MiB | Smooth-line/pass-first presentation |
| <a id="pass-first-normal-speed-mp4"></a>`Pass_First_Normal_Speed.mp4` | 17.17 MiB | Smooth-line/pass-first native recording |
| <a id="smooth-steering-demo-mp4"></a>`Smooth_Steering_Demo.mp4` | 11.36 MiB | Earlier steering presentation |
| <a id="smooth-steering-normal-speed-mp4"></a>`Smooth_Steering_Normal_Speed.mp4` | 14.94 MiB | Earlier steering native recording |
| <a id="trajectory-replay-html"></a>`Trajectory_Replay.html` | 6.32 MiB | Earlier standalone trajectory replay |
| <a id="trajectory-trail-brake-demo-mp4"></a>`Trajectory_Trail_Brake_Demo.mp4` | 11.74 MiB | Trajectory/braking presentation |
| <a id="trajectory-trail-brake-normal-speed-mp4"></a>`Trajectory_Trail_Brake_Normal_Speed.mp4` | 15.52 MiB | Trajectory/braking native recording |
| <a id="apex-demo-jpg"></a>`apex-demo.jpg` | 0.15 MiB | Earlier presentation thumbnail |
| <a id="chase-demo-jpg"></a>`chase-demo.jpg` | 0.15 MiB | Earlier presentation thumbnail |
| <a id="flat-out-test-mp4"></a>`flat-out-test.mp4` | 3.44 MiB | Failed forced-throttle stress test |
| <a id="middle-pass-demo-jpg"></a>`middle-pass-demo.jpg` | 0.14 MiB | Earlier presentation thumbnail |
| <a id="pass-first-demo-jpg"></a>`pass-first-demo.jpg` | 0.15 MiB | Earlier presentation thumbnail |
| <a id="smooth-steering-demo-jpg"></a>`smooth-steering-demo.jpg` | 0.17 MiB | Earlier presentation thumbnail |
| <a id="trajectory-trail-brake-demo-jpg"></a>`trajectory-trail-brake-demo.jpg` | 0.17 MiB | Earlier presentation thumbnail |

## Restore an archived file locally

Point `RACE_PROJECT_ARCHIVE` to the preserved archive directory in your
development project. It contains the original repository-relative paths:

```sh
export RACE_PROJECT_ARCHIVE=/path/to/race-lidar/archives/project-cleanup-20261118-001
cp "$RACE_PROJECT_ARCHIVE/docs/Apex_Demo.mp4" docs/Apex_Demo.mp4
shasum -a 256 docs/Apex_Demo.mp4
```

Compare the digest with the matching entry in `archive_manifest.json` before
using the restored file. The complete pre-cleanup backup also preserves every
original file, including metadata and working output.

Generated material archived separately includes synthetic CPU-run output,
Python bytecode, pytest caches and Finder metadata. These are reproducible
working products; the actual study/reproduction input bundles remain here.
