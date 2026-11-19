# Rear-grid chase: exact demo source

[Watch the 112-second demo](../../docs/Mach_Jesus_Demo.mp4) ·
[Presentation](../../docs/presentation.pdf) · [File and parameter manifest](manifest.json)

This folder preserves the six exact source overrides used by the new chase demo.
The base study code and CPU caches remain separate and unchanged. Historical base
study export: `ca985876a6539cf4630091d72554a91501ae3eef`.

R20 started **P20 and reached sustained P1 at 325.5 simulated seconds**, passing
all 19 active rivals. Peak measured speed was **206.85 km/h** (about 207 km/h).
The hero recorded zero audited overlap samples, track violations and resets.
The chase stopped with a 34.02 m lead; it did not complete the nominal 20-lap race.

This is a combined engineering change, not a newly fitted model or an isolated
ablation. Original Rich005 seed-7 weights and the original VoxelNeXt detector are
unchanged. A hybrid overlay chooses among measured lane opportunities, requests
the curvature/braking envelope, and logs the original learned outputs. Hero-only
racing guards use less lateral uncertainty inflation, closer following and faster
lane commitment. Recorded F1 pace is ignored for the hero, with a finite 100 m/s
planning ceiling; actual power, grip, corner constraints and braking remain.
Rival settings retain their defaults. All parameters and model hashes are in
[manifest.json](manifest.json).

Files retain their original module paths under [source/](source/). They are
reviewable overrides, not a standalone runnable package. A fresh full simulation
also needs the external models/reference, Webots assets and complete race
configuration. See the [setup guide](../../docs/setup.md) and
[artifact catalog](../../artifacts/downloads.json). Weights, raw arrays, machine
addresses and private runtime receipts are excluded from this folder.
