#!/usr/bin/env bash
# Collect a frozen system baseline for a TY1200 validation run.
# Output: <report_dir>/environment.json (+ raw text alongside).
set -uo pipefail

REPORT_DIR="${1:?usage: collect_system_baseline.sh <report_dir>}"
mkdir -p "$REPORT_DIR"
RAW="$REPORT_DIR/environment_raw.txt"
JSON="$REPORT_DIR/environment.json"

IXSMI="/usr/local/corex/bin/ixsmi"
export LD_LIBRARY_PATH="/usr/local/corex/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

{
  echo "=== date ===";        date -Iseconds
  echo "=== uname ===";       uname -a
  echo "=== cpu ===";         grep -m1 'model name' /proc/cpuinfo; echo "cores: $(nproc)"
  echo "=== mem ===";         free -h
  echo "=== swap ===";        swapon --show || true
  echo "=== disk ===";        df -h / /home /data 2>/dev/null
  echo "=== gpu ===";         "$IXSMI" --query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu --format=csv 2>&1 | head -4
  echo "=== time sync ===";   timedatectl 2>&1 | grep -E 'synchronized|NTP|Time zone'
  echo "=== listening ===";   ss -tln | awk 'NR>1{print $4}' | sed 's/.*://' | sort -n | uniq | tr '\n' ' '; echo
  echo "=== python ===";      .venv/bin/python --version 2>&1 || python3 --version
  echo "=== rosclaw ===";     .venv/bin/rosclaw --version 2>&1 || true
  echo "=== ty1200 pkg ===";  python3 -m pip show rosclaw-ty1200 2>/dev/null | grep -E '^(Name|Version)' || true
  echo "=== modeld ===";      systemctl is-active rosclaw-ty1200-modeld.service 2>&1 || true
  echo "=== provenance ===";  echo "rosclaw source: codeload tarball of ros-claw/rosclaw@main (2026-07-31), NOT a git checkout"
} > "$RAW" 2>&1

python3 - "$RAW" "$JSON" <<'PY'
import json, re, sys
raw_path, json_path = sys.argv[1], sys.argv[2]
raw = open(raw_path, encoding="utf-8", errors="replace").read()
def section(name):
    m = re.search(rf"=== {re.escape(name)} ===\n(.*?)(?=\n=== |\Z)", raw, re.S)
    return m.group(1).strip() if m else ""
env = {
    "captured_at": section("date"),
    "uname": section("uname"),
    "cpu": section("cpu"),
    "memory": section("mem"),
    "disk": section("disk"),
    "gpgpu": section("gpu"),
    "time_sync": section("time sync"),
    "listening_ports": section("listening"),
    "python": section("python"),
    "rosclaw": section("rosclaw"),
    "ty1200_platform": section("ty1200 pkg"),
    "modeld": section("modeld"),
    "provenance": section("provenance"),
}
json.dump(env, open(json_path, "w"), indent=2, ensure_ascii=False)
print(f"wrote {json_path}")
PY
