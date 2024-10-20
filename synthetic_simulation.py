# %%
from matplotlib.cm import get_cmap
from utils import run_lstsq, get_sae_info, get_gemma_sae_params_for_layer, get_l0_closest_to, get_sae_info_by_params
import torch
import argparse


# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
args = argparser.parse_args()
device = args.device

# %%
target_layer = 20
target_l0 = 100

prev_layer = target_layer - 1
width="16k"
def get_all_sae_info(layer, sae_width, target_l0, model="gemma_2_9b", layer_type="res"):
    actual_l0 = get_l0_closest_to(get_gemma_sae_params_for_layer(layer, model=model, layer_type=layer_type)[sae_width], target_l0)
    sae_inf = get_sae_info_by_params(layer=str(layer), sae_width=sae_width, sae_l0=actual_l0, model=model, layer_type=layer_type)
    return sae_inf, actual_l0


resid_sae_info, resid_L0 = get_all_sae_info(target_layer, width, target_l0, model="gemma_2_9b", layer_type="res")
mlp_sae_info, mlp_L0 = get_all_sae_info(target_layer, width, target_l0, model="gemma_2_9b", layer_type="mlp")
att_sae_info, att_L0 = get_all_sae_info(target_layer, width, target_l0, model="gemma_2_9b", layer_type="att")
prev_resid_sae_info, prev_resid_L0 = get_all_sae_info(prev_layer, width, target_l0, model="gemma_2_9b", layer_type="res")
# %%

from einops import rearrange
from utils import run_lstsq, calculate_r_squared
from sae_lens import SAE

layer_type = "res"
sae_name = resid_sae_info.sae_name
print(sae_name)
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=f"gemma-scope-9b-pt-{layer_type}",
    sae_id=sae_name,
    device = device,
)

sae.requires_grad_(False)
# turn gradient off globally
torch.set_grad_enabled(False)
def least_squares(x, y, device):
    residuals, r_squared, solution = run_lstsq(x, y, device=device)
    return residuals, r_squared, solution
# %%

import torch
from tqdm import tqdm

def calculate_r_squared_torch(target_var, input_var):
    """
    Calculate R-squared (coefficient of determination) for multidimensional data using PyTorch.
    
    Parameters:
    target_var (torch.Tensor): Target variable with shape [batch, d_model]
    input_var (torch.Tensor): Input variable with shape [batch, d_model]
    
    Returns:
    torch.Tensor: R-squared value
    """
    # Ensure inputs are PyTorch tensors
    target_var = torch.as_tensor(target_var)
    input_var = torch.as_tensor(input_var)
    
    # Check if shapes match
    if target_var.shape != input_var.shape:
        raise ValueError("Shapes of target_var and input_var must match")
    
    # Flatten the tensors to 1D
    target_flat = target_var.reshape(-1)
    input_flat = input_var.reshape(-1)
    
    # Calculate the mean of the target variable
    target_mean = torch.mean(target_flat)
    
    # Calculate total sum of squares
    ss_total = torch.sum((target_flat - target_mean)**2)
    
    # Calculate residual sum of squares
    ss_residual = torch.sum((target_flat - input_flat)**2)
    
    # Calculate R-squared
    r_squared = 1 - (ss_residual / ss_total)
    
    return r_squared


N_d_point = 50_000
resid_act = resid_sae_info.acts_flattened[:N_d_point]

# What we want is to treat x as SAE(x). Then we compare w/ removing specific sets of features (ie removing them)
# then we have a new reconstruction created by zero-ing some set of features
# We can then do lstsq from (x, error) to see if this is perfect, or changes weirdly w/ different amount of components

x = sae(resid_act.to(device))
# x = resid_act.to(device)
# %%

feature_frequencies, average_feature_values_when_active, average_feature_values_overall = resid_sae_info.get_feature_freqs()
sorted_by_average_feature_value_overall = torch.argsort(average_feature_values_overall)


# %%




# Now ablate a subset of features
# Do an increasing percentage of feature ablations
settings = ["FVU of reconstruction", "FVU of Linear Sum"]

noise_levels_act = [1, 0.6]
noise_levels_rec = [0.6, 1]

for noise_level_act in torch.linspace(0, 4, 5):
    for noise_level_rec in torch.linspace(0, 4, 5):
        noise_levels_act.append(noise_level_act.item())
        noise_levels_rec.append(noise_level_rec.item())

recon_fvus = [[] for _ in noise_levels_act]
linear_fvus = [[] for _ in noise_levels_act]
upper_bound_fvus = [[] for _ in noise_levels_act]

total_d_points = 10
range_number = 100//total_d_points
_, num_features = sae.W_enc.shape
for i in range(0, 51, range_number):
    num_features_to_ablate = int(num_features * i / 100)
    features_mask = torch.ones(num_features).to(device)
    # Mask least common features
    features_mask[sorted_by_average_feature_value_overall[:num_features_to_ablate]] = 0


    for noise_ind, noise_level in tqdm(enumerate(noise_levels_act)):
        noise_level_act = noise_levels_act[noise_ind]
        noise_level_rec = noise_levels_rec[noise_ind]
        noise_rec = torch.randn_like(x)*noise_level_rec
        # noise_rec = torch.zeros_like(x)*noise_level
        noise_act = torch.randn_like(x)*noise_level_act

        x_noised = x + noise_act

        # Remove subset of features
        features = sae.encode(x)
        reconstruction = sae.decode(features * features_mask)
        reconstruction += noise_rec

        error = x_noised - reconstruction

        _, _, linear_solution = run_lstsq(x_noised, error, device=device)
        linear_sum = reconstruction + x_noised @ linear_solution.to(device)

        recon_r_squared = calculate_r_squared_torch(x_noised, reconstruction)
        linear_r_squared = calculate_r_squared_torch(x_noised, linear_sum)
        upper_bound_r_squared = calculate_r_squared_torch(x_noised, sae(x_noised))

        recon_fvus[noise_ind].append(recon_r_squared.item())
        linear_fvus[noise_ind].append(linear_r_squared.item())
        upper_bound_fvus[noise_ind].append(upper_bound_r_squared.item())
# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap

plot_fvu = True
reverse_feature_order = True

# Assuming recon_fvus, linear_fvus, noise_levels_act, noise_levels_rec, and range_number are defined

if plot_fvu:
    recon_fvus_plot = [[1-r for r in rs] for rs in recon_fvus]
    linear_fvus_plot = [[1-r for r in rs] for rs in linear_fvus]
else:
    recon_fvus_plot = recon_fvus
    linear_fvus_plot = linear_fvus

x_ticks = [f"{i*range_number}%" for i in range(len(recon_fvus_plot[0]))]
percentage = 1.0
num_points = int(len(recon_fvus_plot[0]) * percentage)

if reverse_feature_order:
    recon_fvus_plot = [rs[::-1] for rs in recon_fvus_plot]
    linear_fvus_plot = [rs[::-1] for rs in linear_fvus_plot]
    x_ticks = x_ticks[::-1]

fig, axs = plt.subplots(2, 2, figsize=(5.5, 3))
axs = axs.flatten()

linestyles = ['-', '--']
cmap = get_cmap('viridis')
color_step = 1.0 / 7
colors = [cmap(1 * color_step), cmap(3 * color_step), cmap(5 * color_step), cmap(7 * color_step)]

line_color_one = "red"
line_color_two = "green"
line_color_three = colors[0]
line_color_four = colors[1]

for noise_ind in range(3):
    # if noise_ind > 1:
    if noise_ind > 2:
        ax = axs[noise_ind]
        data_1 = np.array(recon_fvus_plot[2:])[:, -1]
        data_2 = np.array(linear_fvus_plot[2:])[:, -1]

        linear_estimate = data_1 - data_2
        nonlinear_estimate = data_2

        x = torch.linspace(0, 4, 5)
        y = torch.linspace(0, 4, 5)

        linear_estimate = np.array(linear_estimate).reshape(5, 5)
        nonlinear_estimate = np.array(nonlinear_estimate).reshape(5, 5)

        if noise_ind == 2:
            im = ax.imshow(nonlinear_estimate, cmap='viridis', aspect='auto')
        else:
            im = ax.imshow(linear_estimate, cmap='viridis', aspect='auto')

        ax.set_xticks(range(5))
        ax.set_xticklabels([a.item() for a in x], fontsize=7)
        ax.set_yticks(range(5))
        ax.set_yticklabels([a.item() for a in y], fontsize=7)

        ax.set_xlabel("True Nonlinear Noise Var", fontsize=8)
        ax.set_ylabel("True Linear Noise Var", fontsize=8)

        if noise_ind == 3:
            bar = plt.colorbar(im, ax=axs[noise_ind], fraction=0.046, pad=0.04)
            ax.set_title("                             (FVU difference)", fontsize=8)
            plt.figtext(0.74, 0.444, "Est Linear Error", fontsize=8, color='g', ha ='right')
        else:
            bar = plt.colorbar(im, ax=axs[noise_ind], fraction=0.046, pad=0.04)
            ax.set_title("                                (FVU difference)", fontsize=8)
            plt.figtext(0.264, 0.444, "Est Nonlinear Error", fontsize=8, color='r', ha ='right')

        # Set size of colorbar text
        bar.ax.yaxis.label.set_fontsize(8)


    else:
        ax = axs[noise_ind]
        recon_line, = ax.plot(recon_fvus_plot[noise_ind], label="SAE Reconstruction", linestyle=linestyles[0], color=line_color_three, linewidth=0.8)
        linear_line, = ax.plot(linear_fvus_plot[noise_ind], label="SAE Reconstruction + Error Prediction", linestyle=linestyles[1], color=line_color_four, linewidth=0.8)
        
        last_index = len(recon_fvus_plot[noise_ind]) - 1
        recon_last = recon_fvus_plot[noise_ind][-1]
        linear_last = linear_fvus_plot[noise_ind][-1]
        
        ax.axhline(y=recon_last, color=line_color_three, linestyle='--', linewidth=0.8)
        
        arrow_adjust = 0.5
        arrow_start_x = last_index - arrow_adjust
        arrow_end_x = last_index - arrow_adjust
        
        ax.annotate('',
                    xy=(arrow_start_x, linear_last),
                    xytext=(arrow_end_x, recon_last),
                    arrowprops=dict(arrowstyle='<->', color=line_color_two, shrinkA=0, shrinkB=0, relpos=(0.0, 0.0), mutation_scale=10, lw=0.8),
                    va='center')
        
        ax.annotate('',
                    xy=(arrow_start_x, 0),
                    xytext=(arrow_end_x, linear_last),
                    arrowprops=dict(arrowstyle='<->', color=line_color_one, shrinkA=0, shrinkB=0, relpos=(0.0, 0.0), mutation_scale=10, lw=0.8),
                    va='center')
        
        x_shift = -1.1
        
        if noise_ind == 1:
            ax.text(arrow_end_x - 0.9 + x_shift + 0.08, (recon_last + linear_last) / 2 + 0.01, 'Linear Error$\\approx$',
                    va='center', ha='left', color=line_color_two, fontsize=5.5)
            ax.text(arrow_end_x - 1.0 + x_shift + 0.08, (recon_last + linear_last) / 2 -0.01, 'f(Activation Noise)',
                    va='center', ha='left', color=line_color_two, fontsize=5)
            ax.text(arrow_end_x - 1.15 + x_shift, linear_last / 2 + 0.01, 'Nonlinear Error$\\approx$',
                    va='center', ha='left', color=line_color_one, fontsize=5.5)
            ax.text(arrow_end_x - 1.25 + x_shift, linear_last / 2-0.01, 'f(Reconstruction Noise)',
                    va='center', ha='left', color=line_color_one, fontsize=5)
        else:
            ax.text(arrow_end_x - 2.5 + x_shift, (recon_last + linear_last) / 2, 'Linear Error$\\approx$',
                    va='center', ha='left', color=line_color_two, fontsize=5.5)
            ax.text(arrow_end_x - 1.0 + x_shift, (recon_last + linear_last) / 2, 'f(Activation Noise)',
                    va='center', ha='left', color=line_color_two, fontsize=5)
            ax.text(arrow_end_x - 1.15 + x_shift, linear_last / 2 + 0.005, 'Nonlinear Error$\\approx$',
                    va='center', ha='left', color=line_color_one, fontsize=5.5)
            ax.text(arrow_end_x - 1.25 + x_shift, linear_last / 2-0.015, 'f(Reconstruction Noise)',
                    va='center', ha='left', color=line_color_one, fontsize=5)
                        

        
        ax.set_xticks(range(num_points))
        ax.set_xticklabels(x_ticks, fontsize=7)
        ax.set_ylim(0, 0.16)
        ax.set_xlim(0, num_points-1)
        ax.set_yticks(np.arange(0, 0.16, 0.1))
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(f"$Sae(x')$ Noise Var = { noise_levels_rec[noise_ind]}, $x'$ Noise Var = {noise_levels_act[noise_ind]}", fontsize=8)
        ax.set_xlabel("% features ablated", fontsize=8)
        ax.set_ylabel("FVU", fontsize=8)

plt.tight_layout()
plt.savefig("plots/noise_effect_analysis.pdf", bbox_inches='tight', pad_inches=0.04)
plt.show()
plt.close()

# %%

# Function to calculate correlation
def correlation(x, y):
    return np.corrcoef(x.numpy(), y.numpy())[0, 1]

act_noise = noise_levels_act[2:]
rec_noise = noise_levels_rec[2:]
linear_errors_flat = linear_estimate.flatten()
nonlinear_errors_flat = nonlinear_estimate.flatten()


# Calculate the 2x2 correlation matrix
correlation_matrix = torch.zeros(2, 2)
correlation_matrix[0, 0] = correlation(act_noise, linear_errors_flat)
correlation_matrix[0, 1] = correlation(act_noise, nonlinear_errors_flat)
correlation_matrix[1, 0] = correlation(rec_noise, linear_errors_flat)
correlation_matrix[1, 1] = correlation(rec_noise, nonlinear_errors_flat)

# Labels for rows and columns
row_labels = ['Activation Noise', 'SAE-Reconstruction Noise']
col_labels = ['Linear Error', 'Nonlinear Error']

# Print the correlation matrix
print("2x2 Correlation Matrix:")
print(f"{'':12} {col_labels[0]:15} {col_labels[1]:15}")
for i, row_label in enumerate(row_labels):
    print(f"{row_label:12} {correlation_matrix[i, 0]:15.4f} {correlation_matrix[i, 1]:15.4f}")
