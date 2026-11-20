# Report build

This is a revised copy of the source-verified course report. It adds the
confidence-swept simulated DEV BEV AP results and records the actual CARLA
setup outcome. The earlier report and experimental inputs are preserved.

Compile with Tectonic 0.17.0 from this directory:

```sh
tectonic --keep-logs --untrusted main.tex
```

The checked build has eight pages and uses vector PDF figures. The first build
may need network access to populate the Tectonic bundle cache. If it is already
cached, add `--only-cached` for an offline rebuild.

The report links to the public repository and the 112-second GitHub demo as an
interim video reference. **A YouTube or shareable Google Drive link is still
required for the course submission.** Replace the demo hyperlink and pending
text after the actual hosted link is available; verify viewing access before
submitting. Do not claim the GitHub video satisfies that hosting requirement.

The BEV AP figures are fixed-postprocessing results on previously inspected
simulated development frames. They are not official benchmark AP or evidence
of improved closed-loop driving. CARLA startup observations are infrastructure
results and do not constitute a learned-policy evaluation.
