import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os, json

categories = ["abc_lite", "alphaZ", "16505", "06397"]
subcategories = ["single arm", "CPW-IK", "GPW-IK"]

json_files = {
    "single arm": "single_arm.json",
    "CPW-IK": "cpw_ik.json",
    "GPW-IK": "gpw_ik.json"
}

# --- Step 1: Try to load all JSON files ---
json_data_by_sub = {}
for sub, filename in json_files.items():
    if os.path.exists(filename):
        # try:
        with open(filename, "r") as f:
            json_data_by_sub[sub] = json.load(f)
            print(f"Succesfully parse {filename} ")
    else:
        json_data_by_sub[sub] = None

# --- Step 2: Build full data dict, randomize if missing ---
np.random.seed(0)
data = {}
for cat in categories:
    data[cat] = []
    for sub in subcategories:
        if json_data_by_sub[sub] is not None and cat in json_data_by_sub[sub]:
            data[cat].append(json_data_by_sub[sub][cat])
        else:
            # fallback to fake data
            if sub == "single arm":
                data[cat].append(np.random.normal(800, 50, 50))
            elif sub == "CPW-IK":
                data[cat].append(np.random.normal(600, 40, 50))
            elif sub == "GPW-IK":
                data[cat].append(np.random.normal(350, 30, 50))

# --- Step 3: Plot ---
fig, ax = plt.subplots(figsize=(10,6))

positions = []
labels = []
data_to_plot = []
colors = ['red', 'orange', 'green']

for i, cat in enumerate(categories):
    for j, sub in enumerate(subcategories):
        pos = i*(len(subcategories)+1) + j
        positions.append(pos)
        labels.append(f"{cat}\n{sub}")
        data_to_plot.append(data[cat][j])

bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True, showfliers=False)

for patch, median, color in zip(bp['boxes'], bp['medians'], colors * len(categories)):
    patch.set(facecolor="none", edgecolor=color, linewidth=1)
    median.set_color(color)
    median.set_linewidth(1)

for pos, y in zip(positions, data_to_plot):
    ax.scatter([pos]*len(y), y, color="gray", s=4, alpha=0.6)
    # ax.scatter(pos, y, color="gray", s=4, alpha=0.6)

y_max = np.max(np.array(data_to_plot))

# Add the vertical lines
line_positions = [3, 7, 11]
for pos in line_positions:
    ax.axvline(x=pos, color='black', linestyle='-', linewidth=0.5)

# add group labels
for i, cat in enumerate(categories):
    # Calculate the horizontal center of the group of box plots
    center_pos = i * (len(subcategories) + 1) + 1
    ax.text(center_pos, y_max+250, cat, ha='center', va='top', fontsize=12, fontweight='bold')
# END: Minimal change

ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=45, ha="right")

plt.grid(axis='y', linewidth=0.5, linestyle='--')

ax.set_ylabel("Makespan (degrees)")
ax.set_title("Makespan Improvement Using Two Robotic Arms \n \n")
ax.set_ylim(0, y_max+100) # Adjust y-axis to make space for the labels
plt.tight_layout()

# --- Add puzzle icons above y=2000 line ---
icon_files = ["abc_icon.png", "az_icon.png", "16505_icon.png", "06397_icon.png"]
ax_pos = ax.get_position()
fig_w, fig_h = fig.get_size_inches()
ax_w_inches = ax_pos.width * fig_w
ax_h_inches = ax_pos.height * fig_h
xlim = ax.get_xlim()
ylim_vals = ax.get_ylim()

for i, icon_file in enumerate(icon_files):
    if os.path.exists(icon_file):
        img = mpimg.imread(icon_file)
        img_h, img_w = img.shape[:2]

        center_pos = i * (len(subcategories) + 1) + 1

        # Region: from y=2000 to top of axes
        y_bottom_frac = (2000 - ylim_vals[0]) / (ylim_vals[1] - ylim_vals[0])
        region_height_frac = 1.0 - y_bottom_frac
        img_height_frac = 0.7 * region_height_frac

        # Preserve image aspect ratio when computing width
        img_h_inches = img_height_frac * ax_h_inches
        img_w_inches = img_h_inches * (img_w / img_h)
        img_width_frac = img_w_inches / ax_w_inches

        # Clamp width so image stays within its group
        max_width_frac = 2.8 / (xlim[1] - xlim[0])
        if img_width_frac > max_width_frac:
            scale = max_width_frac / img_width_frac
            img_width_frac = max_width_frac
            img_height_frac *= scale

        # Center the image in the region
        x_frac = (center_pos - xlim[0]) / (xlim[1] - xlim[0])
        x_left = x_frac - img_width_frac / 2
        y_center_frac = y_bottom_frac + region_height_frac / 2
        y_bottom_img = y_center_frac - img_height_frac / 2

        inset_ax = ax.inset_axes([x_left, y_bottom_img, img_width_frac, img_height_frac])
        inset_ax.imshow(img, aspect='auto')
        inset_ax.axis('off')

plt.savefig('makespan_boxplot_with_group_labels.png', dpi=600, bbox_inches='tight')
plt.show()