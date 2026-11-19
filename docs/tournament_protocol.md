# Physical multi-car policy comparison

Each entrant is an independent rigid-body Webots car with its own LiDAR stream,
vision detections, driver state, and steering/throttle/brake commands. A shared
batched detector may combine inference requests; it must preserve request/car
identity. Simulator opponent poses are reserved for independent evaluation and
must not enter the driver or its safety guards.

Measure capacity with 2, 4, 8, then 12 active racers before choosing the field.
Record simulation seconds / wall seconds, detector latency and failures, process
memory, GPU utilization and peak allocation, and any missing sensor decisions.
Physical simulation can run slower than real time; report its measured rate.
Do not infer capacity from GPU memory or the number of model copies alone.

## Race format

Compare the V1, V2, and V3 trained neural checkpoints and the reference-controller
baseline. Use equal replicas of each policy: 4 cars means one of each; 8 means
two each; 12 means three each. Replicas measure interaction/grid variability,
not independent training seeds. If capacity only permits 2 cars, use paired
matches that cover every policy under the same rules, with swapped starting
positions for each pair; compare that series separately from a shared field.

All entrants use identical physical cars, tires/contact friction, power, brakes,
speed and acceleration limits, sensor settings, and the same full track and race
clock. The intended common settings are 60 m/s maximum speed, 8 m/s² lateral
budget, friction coefficient 1.5, 20 ms physics, and 200 ms LiDAR/driver decisions.
Record the exact settings and checkpoint hashes in the run artifacts. The models
learned by supervised imitation from one track; this comparison does not establish
reinforcement learning or general ability on other tracks.

Use equal full distributions of grid slots across policies. For four interleaved
policies in a 20-car field, four cyclic roster shifts put every policy in every
slot exactly once. Do not subtract an estimated grid time. A two-car comparison
can balance slots with two reversed heats; reversal alone is insufficient for
the original four-policy field.

Reversal balances mean positions but can leave different full distributions of
starting slots for different policies. Report those distributions and each heat
winner's starting slot. In the observed final experiment, both valid heat winners
started at zero-based grid index 7, using different policies. That observation
limits attribution of the wins to policy alone; additional grid rotations would
be needed to separate policy effects from particular slots and traffic. Keep the
predeclared ranking unchanged and narrow conclusions to the tested schedule.
Current analysis additionally withholds a supported best-policy claim whenever
the full slot distributions differ. Historical saved JSON and frozen source
retain their original analysis; this stricter check is prospective live code.

Cars begin stationary at their assigned, signed negative track progress, spaced
12 m apart. Race time starts simultaneously. They finish only after crossing
the common ordered-progress target `laps * reference_length_m`; a car starting
behind the line must travel the extra grid distance. Each car uses an independent
ordered progress tracker; nearest-point projection alone is insufficient at
nearby sections of track. No position reset or teleport is allowed after spawn.
Interpolate the shared finish crossing between physics steps. End the heat once
all cars finish/fail, or at the common simulation-time cap.

## Interpretation

The race winner is the valid finisher with the lowest common-clock finish time.
DNFs follow finishers, ordered by actual ordered progress. Exact ties share a
position; car IDs only stabilize display order. A finish needs a completed claim,
finite positive time, full ordered distance, no explicit invalid-lap flag, and
zero recorded pose resets. Missing progress cannot verify a finish.

For policies, rank by clean-completion rate, completion rate, mean position, then
mean time among valid finishes, in that order. Every scheduled entrant counts,
including missing results. A clean completion requires zero track-violation
steps, zero conservative car-envelope overlap samples, and zero pose resets.
Envelope overlaps are a conservative audit and should not be described as proven
physical contacts. Missing incident audits are unknown, never clean.

Each entrant's collision-sample count freezes when its own terminal result is
observed (finish, invalid finish, failure, or timeout). An active entrant still
receives a sample when it overlaps a finished cooldown car or a stopped terminal
car. The raw pair-sample counts continue for all physics steps, including overlaps
between terminal cars. They therefore do not sum to the per-car race counts.
This prevents a finisher's subsequent cooldown from changing its clean result.

Finished cars remain physical participants and follow a cooldown target capped
at 20 m/s, subject to the shared speed and traffic guards. They can encounter
still-active backmarkers, create congestion, or change an active car's outcome.
Freezing a finisher's incident count does not remove those interactions. The
reported results therefore describe this cooldown format as well as the initial
racing field; late active cars may face traffic created by earlier finishers.

The analyzer calls a policy “best among tested” only after at least two heats,
equal policy entries, equal full starting-slot distributions, identical explicitly recorded
common rules, complete car results, complete incident audits, and at least one
clean completion by the leading policy. Otherwise it reports an observed leader
and the missing evidence. It retains race winners and fastest valid finishers
even when no policy earns a clean finish, and can report tied leaders. At a
supervisor time limit, an unfinished controller still marked running, or marked
incomplete specifically because `simulation_closed_before_finish`, is shown as
`timeout` in analysis; its original status and race artifact files are preserved.
One field with two reversed grids is a small experiment; retain per-heat standings
and all incidents instead of making a broad claim from the aggregate score.

## Artifact contract

`tournament.json` lists heats and may supply shared `rules`:

```json
{"heats": [{"heat_id": "heat-1", "path": "heat-1"},
           {"heat_id": "heat-2", "path": "heat-2"}]}
```

Each heat contains `config.json`, `cars/<car_id>/result.json`, and optionally an
authoritative supervisor `heat_results.json`. The configuration has `entrants`
with `car_id`, `policy`, `label`, and zero-based `grid_index`, plus a shared `rules`
object containing `reference_length_m`, `laps`, `sensor_period_ms`,
`physics_step_ms`, `max_speed_mps`, `lateral_accel_budget_mps2`,
`friction_coefficient`, and `vehicle_model`. Keep heat IDs, model-specific values,
and grid settings outside shared `rules`; every additional rule is compared too.

Each car result contains `status`, `completed`, `finish_time_s`, `progress_m`,
`valid_lap`, `position_resets_after_spawn`, `track_violation_steps`, and
`collision_samples`. Include `checkpoint_sha256` (or `model_sha256` or
`neural_driver_checkpoint_sha256`) for neural policies; changing checkpoint
under one policy label prevents a best-policy claim. Unknown collision audits
use `null`. The supervisor can
override them with `heat_results.json` containing `collisions_by_car` as an ID to
sample-count mapping. It can also include `cars` or `results` (list or ID mapping),
`wall_seconds`, and `simulation_seconds`. Per-car files take precedence for
controller outcomes; the independent collision audit takes precedence for overlap
counts. All evaluation files are outputs, never controller observations.

Run `python scripts/analyze_tournament.py <tournament-directory>` to write
`results.json`, `results.csv`, and `report.md`. The library exposes
`evaluate_heat(config, car_results, collisions=None) -> list[dict]` and
`summarize_tournament(heats) -> dict`; each heat input has `heat_id`, `config`,
`results`, optional `collisions`, and optional timing fields.
