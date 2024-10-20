# %%

import pickle
import matplotlib.pyplot as plt
from utils import get_gemma_sae_params_for_layer
import numpy as np
from utils import BASE_DIR

def create_plot(ax, plot_type, size, layer, layer_type):
    result_dict = get_gemma_sae_params_for_layer(
        layer, model=f"gemma_2_{size}", layer_type=layer_type
    )
    all_l0s = result_dict["16k"]

    all_loss_pairs = pickle.load(
        open(f"{BASE_DIR}/results/ce_{layer_type}_{layer}_{size}_{plot_type}.pkl", "rb")
    )

    while all_loss_pairs[0] == []:
        all_loss_pairs.pop(0)

    if plot_type == "l0":
        sort_order = sorted(all_l0s)
        all_loss_pairs = sorted(all_loss_pairs, key=lambda x: sort_order.index(x[1]))
        x_labels = sort_order
    else:
        sort_order = ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]
        all_loss_pairs = sorted(all_loss_pairs, key=lambda x: sort_order.index(x[0]))
        x_labels = sort_order

    if plot_type == "l0":
        l0_filter = [11, 20, 36, 68, 138, 310, 427]
        index_filter = [all_l0s.index(l) for l in l0_filter]
        all_loss_pairs = [all_loss_pairs[i] for i in index_filter]
        x_labels = l0_filter
        
    # all_outer_loop_losses.append((width, resid_L0, linear_percent_recovered, noise_percent_recovered, linear_r_squared_percent_recovered, nonlinar_r_squared_percent_recovered, percentage_of_norm_linear, percentage_of_norm_nonlinear))
    '''
    0: width
    1: resid_L0
    2: linear_percent_recovered
    3: noise_percent_recovered
    4: linear_r_squared_percent_recovered
    5: nonlinar_r_squared_percent_recovered
    6: percentage_of_norm_linear
    7: percentage_of_norm_nonlinear
    
    '''
    percent_change_noise = [r[3] for r in all_loss_pairs]
    percent_change_linear = [r[2] for r in all_loss_pairs]
    percent_r_squared_linear = [r[4].cpu()for r in all_loss_pairs]
    percent_r_squared_nonlinear = [r[5].cpu() for r in all_loss_pairs]
    percent_norm_linear = [r[6]*100 for r in all_loss_pairs]
    percent_norm_nonlinear = [r[7]*100 for r in all_loss_pairs]

    # Width of each bar and spacing
    bar_width = 0.25
    spacing = 0

    # X-axis positions for each group of bars
    x = np.arange(len(x_labels))

    colors = plt.cm.viridis(np.linspace(0, 1, 7))

    # Create the bars
    rects1 = ax.bar(
        x - bar_width / 2 - spacing / 2,
        percent_change_linear,
        bar_width,
        label="Reconstruction + Linear Error",
        color=colors[5],
    )
    rects2 = ax.bar(
        x + bar_width / 2 + spacing / 2,
        percent_change_noise,
        bar_width,
        label="Reconstruction + Nonlinear Error",
        color=colors[1],
    )
    # print(percent_r_squared_linear)
    # print(percent_r_squared_nonlinear)
    # print(rect1.get_bbox().bounds)

        # Add dashed lines for R-squared
    ax.plot([], [], color='red', linestyle='-', label='$R^2$')
    for i, (rect1, rect2) in enumerate(zip(rects1, rects2)):
        x1, y1, w1, h1 = rect1.get_bbox().bounds
        x2, y2, w2, h2 = rect2.get_bbox().bounds
        ax.plot([x1, x1+w1], [percent_r_squared_linear[i]]*2, color='red', linestyle='--', linewidth=1)
        ax.plot([x2, x2+w2], [percent_r_squared_nonlinear[i]]*2, color='red', linestyle='--', linewidth=1)

    # Add dashed lines for Norm
    ax.plot([], [], color='blue', linestyle='-', label='Norm / Total SAE Error Norm')
    for i, (rect1, rect2) in enumerate(zip(rects1, rects2)):
        x1, y1, w1, h1 = rect1.get_bbox().bounds
        x2, y2, w2, h2 = rect2.get_bbox().bounds
        ax.plot([x1, x1+w1], [percent_norm_linear[i]]*2, color='blue', linestyle='--', linewidth=1)
        ax.plot([x2, x2+w2], [percent_norm_nonlinear[i]]*2, color='blue', linestyle='--', linewidth=1)


    # Customize the plot
    ax.set_xlabel("SAE Width" if plot_type == "width" else "L0", fontsize=8)
    if plot_type == "width":
        ax.set_ylabel("Percent Recovered When\nReplacing Layer (%)", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha='right')

    ax.tick_params(axis="both", which="major", labelsize=7)

    # Remove yticks if plotting L0
    if plot_type == "l0":
        ax.set_yticks([])
    else:
        ax.set_yticks([0, 25, 50, 75, 100])

    return ax

# Set up the figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.5))

# Create the l0 plot
ax2 = create_plot(ax2, "l0", "9b", 20, "res")
ax2.set_title("Width 16k", fontsize=10)

# Create the width plot
ax1 = create_plot(ax1, "width", "9b", 20, "res")
ax1.set_title("L0 $\\approx$ 60", fontsize=10)

# Reduce spacing between subplots
plt.subplots_adjust(wspace=0.1)

# Add a single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', fontsize=6, ncol=2, bbox_to_anchor=(0.5, 1.3))

# Adjust the layout and save the figure
plt.savefig("plots/compare_ce_loss_combined.pdf", bbox_inches="tight", pad_inches=0.02)
# %%
