import json
import numpy as np
import matplotlib.pyplot as plt

# Example arrays
dynamic = []
static = []
parallel = []

for i in range(1, 1001):
    with open(
        f"paths/outputs/alphaZ/static_coppelia_1000/trajectory_{i}/report.json", "r"
    ) as file:
        dynamic_data = json.load(file)
    with open(
        f"paths/outputs/alphaZ/static_coppelia_1000/trajectory_{i}/report.json", "r"
    ) as file:
        static_data = json.load(file)
    with open(
        f"paths/outputs/alphaZ/parallel_coppelia_1000/trajectory_{i}/report.json", "r"
    ) as file:
        parallel_data = json.load(file)
    dynamic.append(dynamic_data["best_delta"])
    static.append(static_data["best_delta"])
    parallel.append(parallel_data["best_delta"])

print(np.argmin(static))
print("##################")
print(sorted(np.floor(dynamic))[:100])
print("##################")
print(sorted(np.floor(static))[:100])
print("##################")
print(sorted(np.floor(parallel))[:100])
# Create the box plot
plt.boxplot([static, parallel], tick_labels=["static", "parallel"])

# Optional: Add title and labels
plt.title("Box Plot of 3 Arrays")
plt.ylabel("Values")

# Show the plot
plt.show()
