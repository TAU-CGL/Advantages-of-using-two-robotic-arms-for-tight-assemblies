import os
import json
from pathlib import Path

base = Path("paths/key_results/dual_arm")
assemblies = ["alphaZ", "abc_lite", "16505", "06397"]
max_x = 100

chart = {}

for assembly in assemblies:
    makespans = []
    for x in range(1, max_x + 1):
        path = base / assembly / "balanced" / f"trajectory_{x}" / "report.json"
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    if "best_makespan" in data:
                        makespans.append(data["best_makespan"])
            except Exception as e:
                print(f"Warning: could not read {path}: {e}")
    chart[assembly] = makespans

# Save chart.json at the project root
output_path = Path("gpw_ik.json")
with open(output_path, "w") as f:
    json.dump(chart, f, indent=4)

print(f"chart.json written at {output_path.resolve()}")