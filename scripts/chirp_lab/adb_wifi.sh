#!/usr/bin/env bash
# Switch both phones to adb over Wi-Fi so they can be untethered for the
# 5–20 m runs. Run once with USB attached; afterwards use --wifi with
# run_chirp_lab.py (serial becomes <ip>:5555). Serials and IPs come from
# devices.json so there is one place to edit.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
adb="${ADB:-$HOME/Library/Android/sdk/platform-tools/adb}"
port="$(python3 -c "import json,sys; print(json.load(open('$here/devices.json'))['wifi_adb_port'])")"
python3 - "$here/devices.json" <<'PY' | while read -r serial ip; do
import json, sys
for device in json.load(open(sys.argv[1]))["devices"].values():
    print(device["serial"], device["wifi_ip"])
PY
  echo "== $serial -> $ip:$port"
  "$adb" -s "$serial" tcpip "$port"
  sleep 2
  "$adb" connect "$ip:$port"
done
"$adb" devices -l
