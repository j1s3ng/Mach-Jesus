Rich005 contains two actually fitted portable48-input/6-output MLP checkpoints.
The measured physical failure motivating it is Rich004 seed17's engineering
visibility-entry run: repeated lane reversals prevented displacement, followed
by a fixed-guard stop at290.05m before a stopped obstacle at300m. No opponent GT
was supplied to the driver, teacher, or fitting input.

The new supervised trajectory teacher regularizes only near-equal candidates:
within6 cost units of the best bounded4.5s rollout, an emerging actual slew of
more than0.05m retains its sign, with a consistent default side near zero. A
clearly blocked side or a uniquely safer candidate overrides intent. Fresh/aged
covariance cycles include the observed.25→9.13m uncertainty transition. The old
003 trajectory teacher and all003/004 artifacts remain unchanged. Runtime,
perception, vehicle budgets and the common guard were not modified.

Fitting retains003 hard train/validation examples and004 engineering field4
fit/validation examples, adds20,000 synthetic measured covariance/intent scenes,
and fits68 permitted failed-approach observations through25s. The long repeated
stalled tail is excluded. Rich004-s17-field4-diagnostic is a whole-run engineering
evaluation excluded from fitting and checkpoint choice. Root could observe
aggregate development outcomes while fitting; this worker first opened its
measured rows after checkpoint choices. Formal reserved data was not opened.

| Seed | SHA256 | Selected epoch |
|---|---|---:|
|17|d9837d76d29f80368dae8ec8f611aa67b4c3e7885ed5bab51876fbff34afd9da|29|
|7|89bafe77e99d160aa97aab5510f587112a6c461e2cd7383a381a4af429c94989|41|

CPU fitting took62.2s. Maximum learned parameter changes were0.352/0.398;
NumPy/Torch export errors are below9e-7. Warm normalization preserves the
initialized004 functions to1.2e-6. These are updated weights, not renamed copies.

The fitted failure approach's intent agreement rises from004's77.9%/79.4% to
98.5%/98.5% (seeds17/7); that is training agreement, not generalization. On the
separate4896-state engineering field4 replay, one-step guarded speed shortfall
worsens from004's0.325/0.267m/s to005's1.114/1.060m/s. On the same old-teacher
1800-case synthetic evaluation, avoidable swept-envelope violation also worsens
from004's13.59%/15.24% to005's15.90%/16.23%. These are explicit tradeoffs and
approximate teacher proxies, not measured collision rates.

The first actual005 seed17 visibility-entry run passed with verified audit and
zero scored incidents under the same runtime/detector/guard. During matched
sim11.6..17.4s, actual raw lane reversals fall8→0 and lane changes13→0;005 first
crosses the required320m progress at race21.6s, while004 never does. The remaining
hazards and larger fields still determine foundation fitness and promotion.

Reports: training_report.json; ../rich-driver-005-analysis/post_fit_comparison.json;
old_teacher_fresh_safety.json; physical_visibility_comparison.json; and root's
matched rules/spawn/source verification in physical_passing_comparison.json.
