# %%

import matplotlib
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import get_sae_info_by_params, get_gemma_sae_params_for_layer, get_l0_closest_to, run_lstsq, calculate_r_squared, get_all_tokens, SAEInfoObject
from sae_lens import SAE
import transformer_lens
import einops
from typing import List
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.cm import get_cmap
import numpy as np
from transformers import AutoTokenizer
import torch
import argparse
from tqdm import tqdm

torch.set_grad_enabled(False)

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
args = argparser.parse_args()
device = args.device

# %%


layer = 20
size = "9b"
saes = get_gemma_sae_params_for_layer(layer, model=f"gemma_2_{size}")

widths = [f"{w}k" for w in sorted([int(w[:-1]) for w in saes.keys()])[1:]]
widths.append("1m")
goal_l0 = 60

sae_infos: List[SAEInfoObject] = []
for width in widths:
    l0s = saes[width]
    l0 = get_l0_closest_to(l0s, goal_l0)
    print(l0)
    sae_infos.append(get_sae_info_by_params(layer, width, l0, model=f"gemma_2_{size}"))


# %%
cmap = get_cmap('inferno')
color_step = 1.0 / len(sae_infos)

# %%

nonlinear_norms = []
linear_norms = []
total_error_norms = []
for i, sae_info in enumerate(sae_infos):
    x_gpu = sae_info.acts_flattened.to(device)
    y_gpu = sae_info.sae_error_vecs_flattened.to(device)
    res = run_lstsq(x_gpu, y_gpu, device=device)
    unexplained = y_gpu - x_gpu @ res[2].to(device)
    norms = unexplained.norm(dim=-1).cpu()
    
    width = sae_info.sae_name.split("/")[-2].replace("width_", "")
    nonlinear_norms.append(norms)
    linear_norms.append((x_gpu @ res[2].to(device)).norm(dim=-1).cpu())
    total_error_norms.append(sae_info.sae_error_vecs_flattened.norm(dim=-1).cpu())

nonlinear_norms = torch.stack(nonlinear_norms, dim=1)
total_error_norms = torch.stack(total_error_norms, dim=1)
linear_norms = torch.stack(linear_norms, dim=1)


# %%
num_buckets = 1000

# all_r_squareds_fancy = []
all_r_squareds_basic = []
# all_r_squareds_with_nonlinear_percent_increases_fancy = []
all_r_squareds_with_nonlinear_percent_increases_basic = []
for i in range(len(sae_infos)):
    # r_squareds_fancy = []
    # r_squareds_with_nonlinear_diffs_fancy = []
    r_squareds_basic = []
    r_squareds_with_nonlinear_diffs_basic = []
    for j in tqdm(range(len(sae_infos))):
        # x = create_equal_length_bucket_encoding(total_error_norms[:, i], num_buckets)
        # x_with_ones = torch.cat([x, total_error_norms[:, i:i+1], torch.ones_like(x[:, :1])], dim=1)
        # x_with_ones = x_with_ones[:, x_with_ones.sum(dim=0) > 10]
        # torch.manual_seed(1)
        # r_squared = run_lstsq(x_with_ones, y)[1]
        # r_squareds_fancy.append(r_squared)

        y = total_error_norms[:, j]

        x_with_ones = torch.cat([total_error_norms[:, i:i+1], torch.ones_like(total_error_norms[:, :1])], dim=1)
        torch.manual_seed(1)
        r_squared = run_lstsq(x_with_ones, y)[1]
        r_squareds_basic.append(r_squared)

        # x_1 = create_equal_length_bucket_encoding(total_error_norms[:, i], num_buckets)
        # x_2 = create_equal_length_bucket_encoding(nonlinear_norms[:, i], num_buckets)
        # y = total_error_norms[:, j]
        # x_with_ones = torch.cat([x_1, x_2, total_error_norms[:, i:i+1], nonlinear_norms[:, i:i+1], torch.ones_like(x_1[:, :1])], dim=1)
        # x_with_ones = x_with_ones[:, x_with_ones.sum(dim=0) > 10]
        # torch.manual_seed(1)
        # r_squared = run_lstsq(x_with_ones, y)[1]
        # r_squared_orig = r_squareds_fancy[-1]
        # if r_squared_orig == 1:
        #     r_squareds_with_nonlinear_diffs_fancy.append(0)
        # else:
        #     r_squareds_with_nonlinear_diffs_fancy.append((r_squared - r_squared_orig) / (1 - r_squared_orig) * 100)

        x_with_ones = torch.cat([total_error_norms[:, i:i+1], nonlinear_norms[:, i:i+1], torch.ones_like(nonlinear_norms[:, :1])], dim=1)
        torch.manual_seed(1)
        r_squared = run_lstsq(x_with_ones, y)[1]
        r_squared_orig = r_squareds_basic[-1]
        if r_squared_orig == 1:
            r_squareds_with_nonlinear_diffs_basic.append(0)
        else:
            r_squareds_with_nonlinear_diffs_basic.append((r_squared - r_squared_orig) / (1 - r_squared_orig) * 100)

        # print(r_squareds_with_nonlinear_diffs_basic[-1], r_squareds_with_nonlinear_diffs_fancy[-1], r_squareds_basic[-1], r_squareds_fancy[-1])

    # all_r_squareds_fancy.append(r_squareds_fancy)
    # all_r_squareds_with_nonlinear_percent_increases_fancy.append(r_squareds_with_nonlinear_diffs_fancy)
    all_r_squareds_basic.append(r_squareds_basic)
    all_r_squareds_with_nonlinear_percent_increases_basic.append(r_squareds_with_nonlinear_diffs_basic)

# for all_r_squareds in [all_r_squareds_fancy, all_r_squareds_basic, all_r_squareds_with_nonlinear_percent_increases_fancy, all_r_squareds_with_nonlinear_percent_increases_basic]:
#     all_r_squareds_nan_below_diag = np.array(all_r_squareds)
#     all_r_squareds_nan_below_diag[np.tril_indices(all_r_squareds_nan_below_diag.shape[0], k=-1)] = np.nan
#     plt.imshow(all_r_squareds_nan_below_diag)
#     plt.colorbar()
#     plt.show()

# %%

all_r_squareds_nan_below_diag = np.array(all_r_squareds_basic)
all_r_squareds_nan_below_diag[np.tril_indices(all_r_squareds_nan_below_diag.shape[0], k=-1)] = np.nan

ax, fig = plt.subplots(figsize=(2.75, 2))

plt.imshow(all_r_squareds_nan_below_diag)
bar = plt.colorbar()

# Set bar font size
bar.ax.tick_params(labelsize=8)

# Label the axes
plt.xticks(range(len(sae_infos)), [sae_info.sae_name.split("/")[-2].replace("width_", "") for sae_info in sae_infos], rotation=45, fontsize=7)
plt.yticks(range(len(sae_infos)), [sae_info.sae_name.split("/")[-2].replace("width_", "") for sae_info in sae_infos], fontsize=7)

plt.ylabel("SAE Width (x)", fontsize=7)
plt.xlabel("SAE Width (y)", fontsize=7)

plt.savefig("plots/sae_per_token_predictability.pdf", bbox_inches='tight', pad_inches=0.02)

# %%


all_r_squareds_nan_below_diag = np.array(all_r_squareds_with_nonlinear_percent_increases_basic)
all_r_squareds_nan_below_diag[np.tril_indices(all_r_squareds_nan_below_diag.shape[0], k=-1)] = np.nan

ax, fig = plt.subplots(figsize=(2.75, 2))

plt.imshow(all_r_squareds_nan_below_diag)
bar = plt.colorbar()

# Set bar font size
bar.ax.tick_params(labelsize=8)

# Label the axes
plt.xticks(range(len(sae_infos)), [sae_info.sae_name.split("/")[-2].replace("width_", "") for sae_info in sae_infos], rotation=45, fontsize=7)
plt.yticks(range(len(sae_infos)), [sae_info.sae_name.split("/")[-2].replace("width_", "") for sae_info in sae_infos], fontsize=7)

plt.ylabel("SAE Width (x)", fontsize=7)
plt.xlabel("SAE Width (y)", fontsize=7)

plt.savefig("plots/sae_per_token_predictability_with_nonlinear_diff.pdf", bbox_inches='tight', pad_inches=0.02)

# %%
token_start = 10092
token_end = token_start + 100

average_nonlinear_error = nonlinear_norms.mean(dim=1)

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2), gridspec_kw={'width_ratios': [2, 1]})

x_labels = range(token_start - token_start, token_end - token_start)
token_labels = sae_infos[0].tokens_flattened[token_start:token_end]

# Detokenize
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
token_labels = [f"'{tokenizer.decode(int(t))}'" for t in token_labels]

# Plot data on both axes
for i, info in enumerate(sae_infos):
    color = cmap(i * color_step)
    width = info.sae_name.split("/")[-2].replace("width_", "")
    ax1.plot(x_labels, total_error_norms[token_start:token_end, i],
             label=f"{width} SAE error", 
             color=color,
             linewidth=0.8)
    ax2.plot(x_labels, total_error_norms[token_start:token_end, i],
             color=color,
             linewidth=0.8)
    
# ax1.plot(x_labels, nonlinear_norms[token_start: token_end].min(dim=-1).values, label="Nonlinear error range", color="red", linestyle="--", linewidth=1)
# ax1.plot(x_labels, nonlinear_norms[token_start: token_end].max(dim=-1).values, color="red", linestyle="--", linewidth=1)
# Add shading between the lines
# for ax in [ax1, ax2]:
#     ax.fill_between(x_labels, 
#                     nonlinear_norms[token_start: token_end].min(dim=-1).values, 
#                     nonlinear_norms[token_start: token_end].max(dim=-1).values, 
#                     color="red", alpha=0.2)
# ax1.plot(x_labels, average_nonlinear_error[token_start: token_end], label="Average nonlinear error", color="red", linestyle="--", linewidth=1)
# ax2.plot(x_labels, average_nonlinear_error[token_start: token_end], color="red", linestyle="--", linewidth=1)

ax1.set_xlabel("Token", fontsize=8)
ax1.set_ylabel("L2 Norm", fontsize=8)

stride = 20
y_labels = range(55, 130)
ax1.set_xticks(list(x_labels[::stride]) + [token_end - token_start], list(x_labels[::stride]) + [token_end - token_start], fontsize=7)
ax1.set_yticks(y_labels[::stride], y_labels[::stride], fontsize=7)

ax2.set_xticks(x_labels, token_labels, rotation=70, fontsize=7)
ax2.set_yticks(y_labels[::stride][1:-2], y_labels[::stride][1:-2], fontsize=7)

# Plot legend above the main plot
fig.legend(bbox_to_anchor=(0.07, 0.93), loc='lower left', ncol=len(sae_infos) // 2 + 1, prop={'size': 6.5})

# Set the range for the magnified plot (adjust as needed)
magnify_start = 85
magnify_end = 95
ax2.set_xlim(magnify_start, magnify_end)
top_y = 110
bottom_y = 60
ax2.set_ylim(bottom_y, top_y)

# Add a rectangle patch to show the magnified area
rect = Rectangle((magnify_start, bottom_y), magnify_end - magnify_start, 
                 top_y - bottom_y, fill=False, ec='gray', lw=1)
ax1.add_patch(rect)

# Create connection lines
con1 = ConnectionPatch(xyA=(magnify_start, bottom_y), xyB=(0, 0), coordsA="data", coordsB="axes fraction",
                       axesA=ax1, axesB=ax2, color="gray", linestyle="--", alpha=0.5)
con2 = ConnectionPatch(xyA=(magnify_end, bottom_y), xyB=(1, 0), coordsA="data", coordsB="axes fraction",
                       axesA=ax1, axesB=ax2, color="gray", linestyle="--", alpha=0.5)
con3 = ConnectionPatch(xyA=(magnify_start, top_y), xyB=(0, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax1, axesB=ax2, color="gray", linestyle="--", alpha=0.5)
con4 = ConnectionPatch(xyA=(magnify_end, top_y), xyB=(1, 1), coordsA="data", coordsB="axes fraction",
                       axesA=ax1, axesB=ax2, color="gray", linestyle="--", alpha=0.5)

# Add connection lines to the plot
fig.add_artist(con1)
fig.add_artist(con2)
fig.add_artist(con3)
fig.add_artist(con4)

plt.subplots_adjust(wspace=0.175, hspace=0)

plt.savefig("plots/sae_power_laws_per_token_with_magnification.pdf", bbox_inches='tight', pad_inches=0.02)
plt.show()
plt.close()

# %%

fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(5.5, 2), sharey=True)

for i, sae_info in enumerate(sae_infos):
    color = cmap(i * color_step)
    width = sae_info.sae_name.split("/")[-2].replace("width_", "")
    ax1.plot(nonlinear_norms[token_start: token_end, i], color=color, label=width, linewidth=0.6)
    ax2.plot(linear_norms[token_start: token_end, i], color=color, linewidth=0.6)

ax1.set_title("Norm of Non-Linearly Predictable Error", fontsize=8)
ax2.set_title("Norm of Linearly Predictable Error", fontsize=8)

ax1.set_ylim(40, 100)
ax2.set_ylim(40, 100)

ax1.tick_params(axis='both', which='major', labelsize=7)
ax2.tick_params(axis='both', which='major', labelsize=7)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=1, fontsize=7, title="SAE Width", title_fontsize=8, bbox_to_anchor=(1.05, 0.22))

# fig.supxlabel("Token", fontsize=8, y=0.1, x=0.54)
# ax1.set_ylabel("L2 Norm", fontsize=8)
ax2.set_ylabel("L2 Norm", fontsize=8)
ax1.set_xlabel("Token", fontsize=8)
ax2.set_xlabel("Token", fontsize=8)

plt.tight_layout()
plt.savefig("plots/sae_power_laws_per_token_broken_apart.pdf", bbox_inches='tight', pad_inches=0.02)
plt.show()
plt.close()