# Full-lap reference provenance

The modern Yas Marina layout and source pace come from **Lando Norris's 2024 Abu Dhabi qualifying lap 15**, reported as **82.595 s**. Public OpenF1 session `9658` identifies qualifying; its session result confirms Norris in first place. No Formula 1 account or paid feed was used.

The downloaded reference is `.local/f1-reference/reference.npz`, with arrays `time_s`, `x_m`, `y_m`, `speed_mps` and optional gear/throttle/brake channels. `reference.json` preserves URLs, units, the file hash and processing. Raw location, car-data, lap and driver JSON are beside it.

The initial reference contains 325 position-time samples from 0 to 82.595 s, a median interval of 0.24 s, an approximately 5,229.9 m XY polyline and a 1.44 m endpoint gap. The download keeps original position samples within the lap and interpolates its boundaries. It does not force geometric closure. Any smoothing, closure or vehicle-limit changes applied by the simulator belong to the generated track/driver artifact, not the original data.

OpenF1 explicitly describes location as approximate and says it **lacks reliable lateral placement across the track**. This supports a recognizable F1 circuit outline and source speed profile, not a claim that we copied Norris's exact outside/apex/exit line. The simulator computes its own smooth line and speed limits for the available Webots vehicle/tire model. Its lap time is not an F1 performance result.

F1 timing-feed X/Y values are converted from decimetres to metres, consistent with the [FastF1 telemetry data model](https://raw.githubusercontent.com/theOehrly/Fast-F1/master/fastf1/core.py). Speed is converted from km/h to m/s. Coordinates have an arbitrary track-local origin; they are not surveyed/geodetic positions. OpenF1 lap-start timestamps are also approximate.

The initial attempt through current FastF1 could not load Formula 1's account-gated timing endpoint. Acquisition switched to OpenF1's explicitly public historical API, rather than bypassing that access control. FastF1 was installed in a separate `.local/f1-reference/.venv` and the shared workspace environment was not changed.

To reproduce in a fresh output directory using the existing project Python:

```sh
python scripts/fetch_f1_reference.py --output .local/f1-reference-new
```

Only NumPy and Python's standard library are needed for this working OpenF1 fetcher. The public API may be unavailable or rate limited; the helper preserves existing references and requires a new output to reacquire.

Sources: [OpenF1 historical-data access and schema](https://openf1.org/docs/), [location limitations](https://openf1.org/docs/#location), [lap timing](https://openf1.org/docs/#laps), [qualifying session](https://api.openf1.org/v1/sessions?session_key=9658), [Norris lap timings](https://api.openf1.org/v1/laps?session_key=9658&driver_number=4), [qualifying winner](https://api.openf1.org/v1/session_result?session_key=9658&position=1).
