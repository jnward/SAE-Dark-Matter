# %%

from utils import run_lstsq, get_sae_info, get_gemma_sae_params_for_layer, get_l0_closest_to, get_sae_info_by_params
import pickle
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from tqdm import tqdm
from utils import BASE_DIR
import argparse

# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
args = argparser.parse_args()

device = args.device

# %%

size = "9b"

use_acts = True

# %%

# Get list of all SAEs that you can load into SAE infos
sae_params = get_gemma_sae_params_for_layer(layer=20, model=f"gemma_2_{size}")

# %%

arbirary_sae_info = get_sae_info(layer="20", sae_name="layer_20/width_131k/average_l0_10", model=f"gemma_2_{size}")
acts = arbirary_sae_info.acts_flattened
acts_norm_prediction_percent = run_lstsq(acts, acts.norm(dim=-1)**2, device=device)[1]

# %%

sae_width_and_l0_to_norm_prediction_percents = []
for width in tqdm(sae_params.keys()):
    l0s = sae_params[width]
    for l0 in l0s:
        sae_info = get_sae_info_by_params(layer="20", sae_width=width, sae_l0=l0, model=f"gemma_2_{size}")
        x = sae_info.acts_flattened
        y = -sae_info.sae_error_vecs_flattened
        unexplained_noise = y - x @ run_lstsq(x, y, device=device, lstsq_token_threshold="all")[2]

        percents = [] 
        for x in [sae_info.reconstruction_vecs_flattened, -sae_info.sae_error_vecs_flattened, unexplained_noise, sae_info.reconstruction_vecs_flattened+unexplained_noise, -sae_info.sae_error_vecs_flattened-unexplained_noise]:
            for include_ones in [True, False]:
                y = x.norm(dim=-1)**2
                if include_ones:
                    x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
                if use_acts:
                    norm_prediction_percent = run_lstsq(acts, x.norm(dim=-1)**2, device=device, randomize_order=True)[1]
                else:
                    norm_prediction_percent = run_lstsq(x, x.norm(dim=-1)**2, device=device, randomize_order=True)[1]
                percents.append(norm_prediction_percent)
        sae_width_and_l0_to_norm_prediction_percents.append((width, l0, percents))

# %%

if use_acts:
    save_file = f"{BASE_DIR}/data/sae_width_and_l0_to_norm_prediction_percents_gemma_{size}_using_acts.pkl"
else:
    save_file = f"{BASE_DIR}/data/sae_width_and_l0_to_norm_prediction_percents_gemma_{size}.pkl"

os.makedirs(f"{BASE_DIR}/data", exist_ok=True)

to_save = (sae_width_and_l0_to_norm_prediction_percents, acts_norm_prediction_percent)

if os.path.exists(save_file):
    raise ValueError("File already exists")

with open(save_file, "wb") as f:
    pickle.dump(to_save, f)

# %%

# Create the plot
fig, ax = plt.subplots(figsize=(4, 3.5))

legend_elements = []

for use_acts in [True]:
    if use_acts:
        save_file = f"{BASE_DIR}/data/sae_width_and_l0_to_norm_prediction_percents_gemma_{size}_using_acts.pkl"
    else:
        save_file = f"{BASE_DIR}/data/sae_width_and_l0_to_norm_prediction_percents_gemma_{size}.pkl"

    with open(save_file, "rb") as f:
        sae_width_and_l0_to_norm_prediction_percents, acts_norm_prediction_percent = pickle.load(f)

    norm_prediction_names = ["$Sae(x)$", "$Sae Error(x)$", "$NonlinearError(x)$", "$Sae(x)$ -\n $NonlinearError(x)$", "$LinearError(x)$"]
    norm_prediction_vals = [[] for _ in range(len(norm_prediction_names))]
    use_with_bias = False
    for width, l0, percents in sae_width_and_l0_to_norm_prediction_percents:
        for i in range(len(norm_prediction_names)):
            if use_with_bias:
                percent = percents[2 * i]
            else:
                percent = percents[2 * i + 1]
            norm_prediction_vals[i].append(percent)

    norm_prediction_totals = [sum(vals) / len(vals) for vals in norm_prediction_vals]
    norm_prediction_names.append("x")
    norm_prediction_totals.append(acts_norm_prediction_percent)

    norm_prediction_vals.append([acts_norm_prediction_percent])
    order = [5, 0, 3, 1, 4, 2]
    norm_prediction_names = [norm_prediction_names[i] for i in order]
    norm_prediction_totals = [norm_prediction_totals[i] for i in order]
    norm_prediction_vals = [norm_prediction_vals[i] for i in order]

    norm_prediction_means = [np.mean(vals) for vals in norm_prediction_vals]

    # Create violin plots
    parts = ax.violinplot(norm_prediction_vals, showmeans=False, showmedians=False, showextrema=False)

    # Customize violin plots
    for i, pc in enumerate(parts['bodies']):
        color = plt.cm.viridis(i / len(norm_prediction_vals))
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_alpha(0.7 if use_acts else 0.2)
        
        if not use_acts:
            pc.set_hatch('///')

    # Add legend element
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=plt.cm.viridis(0.5), 
                                         alpha=0.7 if use_acts else 0.2,
                                         hatch='///' if not use_acts else None,
                                         label='Predict from x' if use_acts else 'Predict from self'))

    if use_acts:
        plt.ylim(0.55, 1.03)
    else:
        plt.ylim(-0.1, 1.03)

# Customize the plot
ax.set_ylabel('Norm Prediction Test $R^2$', fontsize=10)
ax.set_xticks(np.arange(1, len(norm_prediction_names) + 1))
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_xticklabels(norm_prediction_names, rotation=45, ha='right', fontsize=9)

# Add legend
# ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=8)

# Save the figure
plt.tight_layout()
plt.savefig("plots/norm_prediction_test.pdf", bbox_inches='tight')

plt.show()
plt.close()
# %%
