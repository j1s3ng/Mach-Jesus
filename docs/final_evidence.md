# Final evidence gate

`scripts/final_evidence.py` audits the complete study: the original 15-case
baseline holdout, the frozen finalist's paired 15+15 fresh confirmation, four
balanced championship races, and all six fresh hazard layouts for each of the
four roster policies. Exactly 73 unique, authoritative closed attempts are
required. The immutable joint admission, all 58 final job arguments and output
paths, canonical case inventories, and the engineering-only frozen roster
chain must agree. The one exact historical presentation-demo exception keeps
its existing narrow scope; it cannot replace a required case.

No command trains, queries a detector, promotes a policy, changes the queue,
or rewrites earlier outcomes, progress, reports, caches or ledger history.
Run the actual audit only after the capacity benchmarks and all 73 cases have
closed, using a new output directory inside the campaign:

```sh
python scripts/final_evidence.py \
  --campaign runs/improvement-12h \
  --selection-receipt PATH_TO_FROZEN_SELECTION_RECEIPT \
  --confirmation-pack PATH_TO_IMMUTABLE_CONFIRMATION_PACK \
  --final-pack PATH_TO_IMMUTABLE_FINAL_PACK \
  --joint-receipt PATH_TO_ACTUAL_JOINT_ADMISSION_RECEIPT \
  --joint-forecast PATH_TO_ROOT_APPROVED_FORECAST \
  --championship-results PATH_TO_STRICT_FINAL_ANALYSIS_RESULTS \
  --output-directory runs/improvement-12h/final-evidence-NEW
```

The audit creates isolated baseline and confirmation auditors with decisions
disabled and empty caches. It also reruns the strict 28-case scorer. Full byte
inventories before and after these audits cover all 73 runs' config, results,
controller summaries, control traces, predictions, actions, GT poses,
leaderboard and authoritative contact totals; optional contact-stream absence
is recorded. The inventories include the three frozen bundles, their models
and reference, canonical manifests, selection receipts and evaluator sources.
Derived audit caches and raw LiDAR/video bytes are outside this scoring scope;
media artifacts have separate provenance. Later unrelated capacity ledger
events do not invalidate the selective study fingerprint. Extra attempts in a
required formal arm do invalidate it.

The packaging API is:

```python
from scripts.final_evidence import verify_final_evidence
report = verify_final_evidence(campaign, evidence_report_path,
                               require_complete=True)
```

The read-only verifier rehashes every scored input and bound report, rechecks
the exact 73-case identity and frozen roster, and recomputes public metrics and
flags from the fresh audit reports. It verifies the actual unique ledger
decision-to-promotion chain rather than trusting an incumbent marker. Editing
a completion flag or metric and recalculating the JSON digest is insufficient.

`complete_source_verified_study` means all mandatory observations are present
and independently verified. A genuine DNF or hazard failure remains a complete
observation. `formal_promoted` and `qualified_champion_released` are separate
outcomes governed by their unchanged safety, pace and championship gates.
An incomplete or infrastructure-invalid attempt cannot count as a complete
source-verified study. Review packaging must remain explicitly incomplete
until this gate succeeds.
