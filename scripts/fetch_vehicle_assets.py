"""Fetch the official Webots generic car/tire assets at the installed app commit.

Upstream license notices stay intact. These assets are used only inside Webots.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "c85603f2280c68ebdcc0a4102e43026fcbf2bc71"
BASE = f"https://raw.githubusercontent.com/cyberbotics/webots/{COMMIT}/projects/vehicles/protos/abstract/"


def main():
    output = ROOT / "webots/protos/vehicle"
    output.mkdir(parents=True, exist_ok=True)
    pending = ["Car.proto", "VehicleLights.proto", "VehicleWheel.proto", "AckermannVehicle.proto", "GenericTireAppearance.proto"]
    entries = {}
    while pending:
        relative = pending.pop(0)
        if relative in entries:
            continue
        if ".." in Path(relative).parts or relative.startswith("/"):
            raise ValueError(f"Unexpected upstream asset path: {relative}")
        target = output / relative
        url = BASE + relative
        if not target.exists():
            data = urllib.request.urlopen(url, timeout=30).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        data = target.read_bytes()
        entries[relative] = {"source":url, "sha256":hashlib.sha256(data).hexdigest(), "bytes":len(data)}
        if relative.endswith(".proto"):
            source = data.decode()
            for dependency in re.findall(r'EXTERNPROTO\s+"([^\"]+)"',source):
                if "://" not in dependency:
                    pending.append(str(Path(relative).parent / dependency))
            pending.extend(asset for asset in re.findall(r'"([^\"]+\.(?:png|jpe?g|wav|obj|stl))"', source)
                           if "://" not in asset)
    (output / "sources.json").write_text(json.dumps({"repository":"https://github.com/cyberbotics/webots", "commit":COMMIT,
        "license":"Upstream Webots asset license; see preserved file headers", "files":entries},indent=2)+'\n')
    print(json.dumps({"output":str(output),"files":len(entries),"bytes":sum(x["bytes"] for x in entries.values())},indent=2))


if __name__ == "__main__":
    main()
