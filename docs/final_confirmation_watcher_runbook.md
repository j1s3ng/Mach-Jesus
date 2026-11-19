# Root final-confirmation watcher sequence

This runbook prepares no cases and launches no process. It is the root-owned
sequence after the engineering-selected finalist and complete final58 are ready.

1. Freeze the actual engineering-selected roster/receipt only after the required
   engineering candidate/CEM outcomes are known. Retain the first immutable
   engineering selection proof; do not choose weights from formal results.
2. Pass the actual complete-work joint admission and explicitly append all30
   confirmation plus all28 championship/geometry jobs together through
   `append_joint_final_jobs(..., execute=True)`. Verify its exact receipt and the
   `joint_final58_jobs_enqueued` event before starting the watcher.
3. **Immediately after that successful enqueue**, root launches exactly one
   confirmation watcher with the actual immutable selection receipt and `--act`:

   ```sh
   python scripts/watch_final_confirmation.py \
     --campaign runs/improvement-12h \
     --selection-receipt ACTUAL_FROZEN_SELECTION_RECEIPT.json \
     --output-directory runs/improvement-12h/evaluation-final-confirmation \
     --act
   ```

   Use root's existing durable process supervision and preserve its log/PID
   identity. The command above is a recipe, not an executed launch. Keep the
   watcher source/selection/canonical case files stable. Its `watcher.lock`
   rejects duplicate owners; it rechecks the deadline and unchanged formal gate.
4. Inspect the actual watcher report and ledger. If the baseline bootstrap and
   paired confirmation gate genuinely permit promotion, the act-enabled watcher
   must record the exact `promotion_decision` followed by the unique
   `candidate_promoted` event. Passing derived metrics, a chosen incumbent marker,
   or a saved green report cannot stand in for that actual chain. If the prior
   approved descriptive fallback applies, `--act` remains descriptive and cannot
   initialize an incumbent or promote anything. Preserve negative outcomes.
5. Once baseline15 + confirmation30 + championship4 + geometry24 are all closed,
   run the new final73 audit into a **fresh** output directory. Its internal
   baseline/confirmation auditors deliberately use `act=False`; they verify and
   report an already recorded outcome. They cannot substitute for step3 or create
   a missing promotion event. The fresh73 verifier must still pass immediately
   before final packaging.

No watcher launch is authorized by this document before the actual selection
and exact58 enqueue. A complete source-verified study, actual formal promotion,
and a qualified championship winner are separate report fields. Completion must
remain reportable even when genuine negative outcomes prevent either release.
