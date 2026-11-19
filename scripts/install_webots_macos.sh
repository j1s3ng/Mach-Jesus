#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
version=nightly_12_11_2026
app="$root/.local/Webots.app"
if [[ -x "$app/Contents/MacOS/webots" ]]; then
  "$app/Contents/MacOS/webots" --version
  exit 0
fi
mkdir -p "$root/.local/downloads"
image="$root/.local/downloads/webots-R2025b.dmg"
curl --fail --location --retry 2 --output "$image" "https://github.com/cyberbotics/webots/releases/download/$version/webots-R2025b.dmg"
mount_dir=$(mktemp -d "${TMPDIR:-/tmp}/race-webots.XXXXXX")
trap 'hdiutil detach "$mount_dir" >/dev/null 2>&1 || true; rmdir "$mount_dir" 2>/dev/null || true' EXIT
hdiutil attach -readonly -nobrowse -mountpoint "$mount_dir" "$image"
ditto "$mount_dir/Webots.app" "$app"
"$app/Contents/MacOS/webots" --version
