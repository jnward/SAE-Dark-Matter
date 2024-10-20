# %%

from utils import run_lstsq, get_gemma_sae_params_for_layer, get_l0_closest_to, get_sae_info_by_params
import torch
from einops import rearrange
from tqdm import tqdm
import pickle
import os
import argparse

# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
args = argparser.parse_args()
device = args.device

# %%

def get_all_sae_info(layer, sae_width, target_l0, model="gemma_2_9b", layer_type="res"):
    actual_l0 = get_l0_closest_to(get_gemma_sae_params_for_layer(layer, model=model, layer_type=layer_type)[sae_width], target_l0)
    sae_inf = get_sae_info_by_params(layer=str(layer), sae_width=sae_width, sae_l0=actual_l0, model=model, layer_type=layer_type)
    return sae_inf, actual_l0

torch.set_grad_enabled(False)
def least_squares(x, y, device):
    residuals, r_squared, solution = run_lstsq(x, y, device=device)
    return residuals, r_squared, solution


result_dict = get_gemma_sae_params_for_layer(20, model="gemma_2_9b", layer_type="res")

width = "16k"
target_layer = 20
target_l0 = 60


# First get resid results
resid_sae_info, resid_L0 = get_all_sae_info(target_layer, width, target_l0, model="gemma_2_9b", layer_type="res")
# Now we want the error term from the resid_sae
resid_act = resid_sae_info.acts_flattened
resid_sae_rec = resid_sae_info.reconstruction_vecs_flattened
resid_sae_error = resid_act - resid_sae_rec

_, _, resid_solution = least_squares(resid_act, resid_sae_error, device)
resid_linear_error = resid_act @ resid_solution
resid_nonlinear_error = resid_sae_error - resid_linear_error
assert torch.allclose(resid_act, resid_sae_rec + resid_linear_error + resid_nonlinear_error, atol=1e-5)
assert torch.allclose(resid_sae_error,  resid_linear_error + resid_nonlinear_error, atol=1e-5)

# Now we want to predict SAE-error and nonlinear_error from previous model components
settings = ['res', 'prev_res', 'mlp', 'att']

def circuits_get_acts_and_r_squared_results(settings, target_explanation, verbose=False):
    results_dict = {}
    acts_for_sim_dict = {}
    sae_error_solutions = {}
    for setting in settings:
        if(setting == "prev_res"):
            t_layer = target_layer-1
            sae_info, L0 = get_all_sae_info(t_layer, width, target_l0, model="gemma_2_9b", layer_type="res")
        else:
            sae_info, L0 = get_all_sae_info(target_layer, width, target_l0, model="gemma_2_9b", layer_type=setting)
        act = sae_info.acts_flattened
        sae_rec = sae_info.reconstruction_vecs_flattened
        if(setting == "att"):
            print("attn_shapes")
            print(act.shape, sae_rec.shape)
            # Combine a specific set of activations
            act = rearrange(act, 'b h w -> b (h w)')
            sae_rec = rearrange(sae_rec, 'b h w -> b (h w)')
        sae_error = act - sae_rec
        _, _, solution = least_squares(act, sae_error, device)
        linear_error = act @ solution
        nonlinear_error = sae_error - linear_error

        assert torch.allclose(act, sae_rec + linear_error + nonlinear_error, atol=1e-5)



        # Now we want to predict the target_explanation from each of the components: act, sae_rec, linear_error, and nonlinear_error
        _, r_squared_act, _ = least_squares(act, target_explanation, device)
        _, r_squared_sae_rec, sae_rec_linear_solutions = least_squares(sae_rec, target_explanation, device)
        _, r_squared_linear_error, _ = least_squares(linear_error, target_explanation, device)
        _, r_squared_nonlinear_error, _ = least_squares(nonlinear_error, target_explanation, device)



        # Save the results
        acts_for_sim_dict[setting] = {
            "act": act,
            "sae_rec": sae_rec,
            "linear_error": linear_error,
            "nonlinear_error": nonlinear_error
        }

        results_dict[setting] = {
            "r_squared_act": r_squared_act,
            "r_squared_sae_rec": r_squared_sae_rec,
            "r_squared_linear_error": r_squared_linear_error,
            "r_squared_nonlinear_error": r_squared_nonlinear_error
        }

        sae_error_solutions[setting] = sae_rec_linear_solutions
        if(verbose):
            print(f"Setting: {setting}")
            print(f"R^2 for act: {r_squared_act}")
            print(f"R^2 for sae_rec: {r_squared_sae_rec}")
            print(f"R^2 for linear_error: {r_squared_linear_error}")
            print(f"R^2 for nonlinear_error: {r_squared_nonlinear_error}")
            print("\n")
    return results_dict, acts_for_sim_dict, sae_error_solutions


sae_error_results_dict, acts_for_sim_dict, sae_error_linear_solutions = circuits_get_acts_and_r_squared_results(settings, target_explanation=resid_sae_error, verbose=True)
nonlinear_error_results_dict, acts_for_sim_dict, nonlinear_error_linear_solutions = circuits_get_acts_and_r_squared_results(settings, target_explanation=resid_nonlinear_error, verbose=True)
linear_error_results_dict, acts_for_sim_dict, linear_error_linear_solutions = circuits_get_acts_and_r_squared_results(settings, target_explanation=resid_linear_error, verbose=True)

# %%

# Save all of the results
if not os.path.exists("data/simple_circuits_results.pkl"):
    pickle.dump((sae_error_results_dict, nonlinear_error_results_dict, linear_error_results_dict), open("data/simple_circuits_results.pkl", "wb"))

# %%

# Load all of the results
sae_error_results_dict, nonlinear_error_results_dict, linear_error_results_dict = pickle.load(open("data/simple_circuits_results.pkl", "rb"))

# %%

# Now we'd like to plot a bar-plot of the R^2 values
import matplotlib.pyplot as plt

def plot_r_squared_bar_plot(results_dict, title):
    # Plot each group of bars by their settings
    setting_labels = ["x", "sae_rec", "linear_error", "nonlinear_error"]
    settings = ["res", "prev_res", "mlp", "att"]
    fig, ax = plt.subplots()
    bar_width = 0.2
    bar_positions = torch.arange(4)
    for i, setting in enumerate(settings):
        r_squared_values = [results_dict[setting][key] for key in results_dict[setting]]
        ax.bar(bar_positions + i*bar_width, r_squared_values, bar_width, label=settings[i])
    # label y axis as R^2
    ax.set_ylabel("R^2")
    # ax.set_title(title)
    ax.set_xticks(bar_positions + 1.5*bar_width)
    ax.set_xticklabels(setting_labels)
    ax.legend()
    # Set x-labels
    plt.show()


def plot_sae_rec_r_squared_bar_plot(results_dict, title):
    settings = ["res", "prev_res", "mlp", "att"]
    sae_rec_values = [results_dict[setting]["r_squared_sae_rec"] for setting in settings]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_positions = range(len(settings))
    
    # Plot bars
    ax.bar(bar_positions, sae_rec_values)
    
    # Customize the plot
    ax.set_ylabel("R²")
    # ax.set_title(title)
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(settings, rotation=45, ha='right')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import torch
from matplotlib.cm import get_cmap

cmap = get_cmap("viridis")
color_step = 1.0 / 7
color = cmap(3 * color_step)

def plot_combined_r_squared_bar_plot(nonlinear_error_dict, sae_error_dict, linear_error_dict, main_title):
    settings = ["res", "prev_res", "mlp", "att"]
    labels = ["residual", "prev residual", "MLP", "attention"]
    error_types = ["Nonlinear Error", "SAE-Error", "Linear Error"]
    results_dicts = [nonlinear_error_dict, sae_error_dict, linear_error_dict]

    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 2 / 3, 1.5), sharey=True)
    # fig.suptitle(main_title, fontsize=10)

    for i, (ax, error_type, results_dict) in enumerate(zip(axes, error_types, results_dicts)):
        sae_rec_values = [results_dict[setting]["r_squared_sae_rec"] for setting in settings]
        
        bars = ax.bar(range(len(settings)), sae_rec_values, color=color)
        
        ax.set_ylabel("R²" if i == 0 else "")
        ax.set_title(f"Predicting\n{error_type}", fontsize=7)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 0.62)
        ax.set_yticklabels([f'{y:.2f}' for y in ax.get_yticks()], fontsize=6)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=6)
   
    plt.savefig("plots/predicting_from_recons.pdf", bbox_inches="tight")
    plt.show()


# Example usage:
plot_combined_r_squared_bar_plot(
    nonlinear_error_results_dict,
    sae_error_results_dict,
    linear_error_results_dict,
    "R² Values for Predicting Errors from SAE Reconstructions"
)