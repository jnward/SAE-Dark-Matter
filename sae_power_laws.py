# %%

import matplotlib
from matplotlib.patches import ConnectionPatch
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import get_sae_info_by_params, get_gemma_sae_params_for_layer, get_l0_closest_to, run_lstsq, fraction_variance_unexplained, BASE_DIR
import pickle
from scipy.interpolate import griddata
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from matplotlib.cm import get_cmap
from utils import normalized_mse
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import argparse
torch.set_grad_enabled(False)

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--to_plot", choices=["both", "pursuit"], default="both")
args = argparser.parse_args()

device = args.device
to_plot = args.to_plot
layer = 20
size = "9b"


# %%

os.makedirs(f"{BASE_DIR}/data", exist_ok=True)
save_file = f"{BASE_DIR}/data/sae_power_laws_{size}.pkl"

if not os.path.exists(save_file):

    saes = get_gemma_sae_params_for_layer(layer, model=f"gemma_2_{size}")

    saes = list(saes.items())
    normalized_mses = [[]]
    normalized_mses_with_predictions = [[]]
    fvus = [[]]
    fvus_with_predictions = [[]]
    sae_error_norm_r_squareds = [[]]
    sae_error_vec_r_squareds = [[]]
    empirical_l0s = [[]]

    for width, l0s in saes:
        print(width, l0s)
        for l0 in l0s:
            try:
                sae_info = get_sae_info_by_params(layer, width, l0, num_cols_start=1, model=f"gemma_2_{size}")
            except FileNotFoundError:
                print(f"Could not find {width} {l0}")
                continue

            print(f"Found {width} {l0}")

            normalized_mses[-1].append(normalized_mse(sae_info.reconstruction_vecs_flattened, sae_info.acts_flattened))
            fvus[-1].append(fraction_variance_unexplained(sae_info.acts_flattened, sae_info.reconstruction_vecs_flattened))

            empirical_l0 = 0
            for active_feature_instance in sae_info.active_sae_features:
                num_zero_indices = (active_feature_instance[0] == 0).sum()
                empirical_l0 += len(active_feature_instance[0]) - num_zero_indices.item()
            empirical_l0 /= len(sae_info.active_sae_features) * 1023
            empirical_l0s[-1].append(empirical_l0)

            x_gpu = sae_info.acts_flattened.to(device)
            y_gpu = -sae_info.sae_error_vecs_flattened.to(device)
            _, r_squared, sol = run_lstsq(x_gpu, y_gpu, device=device)
            sae_error_vec_r_squareds[-1].append(r_squared)

            predictions = x_gpu @ sol.to(device)
            normalized_mses_with_predictions[-1].append(normalized_mse(sae_info.reconstruction_vecs_flattened + predictions.to("cpu"), sae_info.acts_flattened))
            fvus_with_predictions[-1].append(fraction_variance_unexplained(sae_info.acts_flattened, sae_info.reconstruction_vecs_flattened + predictions.to("cpu")))

            y_gpu = y_gpu.norm(dim=-1) ** 2
            _, r_squared, _ = run_lstsq(x_gpu, y_gpu, device=device)
            sae_error_norm_r_squareds[-1].append(r_squared)

        normalized_mses.append([])
        fvus.append([])
        normalized_mses_with_predictions.append([])
        fvus_with_predictions.append([])
        empirical_l0s.append([])
        sae_error_norm_r_squareds.append([])
        sae_error_vec_r_squareds.append([])

    to_save = (saes, normalized_mses[:-1], normalized_mses_with_predictions[:-1], fvus[:-1], fvus_with_predictions[:-1], empirical_l0s[:-1], sae_error_norm_r_squareds[:-1], sae_error_vec_r_squareds[:-1])

    with open(save_file, "wb") as f:
        pickle.dump(to_save, f)

# %%

# Load in grad pursuit date

off_by_limit = 0.2

target_l0 = 60
print("Target L0:", target_l0)

normalized_mses_to_plot_grad_pursuit = []
normalized_mse_with_preds_to_plot_grad_pursuit = []
fvus_to_plot_grad_pursuit = []
fvus_with_preds_to_plot_grad_pursuit = []

all_sae_params = get_gemma_sae_params_for_layer(20, model=f"gemma_2_{size}")
acts = get_sae_info_by_params(20, "16k", min(all_sae_params["16k"]), model=f"gemma_2_{size}").acts_flattened

sae_widths = ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]

for width in sae_widths:
    print(width)
    l0s = all_sae_params[width]
    closest_l0 = get_l0_closest_to(l0s, target_l0)
    sae_name_no_slashes = f"layer_20_width_{width}_average_l0_{closest_l0}"
    acts_gpu = acts.to(device)
    path = f"{BASE_DIR}/data/grad_pursuit_reconstructions_{sae_name_no_slashes}-{sae_name_no_slashes}.pt"
    if os.path.exists(path):
        recons = torch.load(path).to(device)
    else:
        raise ValueError("File not found")

    normalized_mses_to_plot_grad_pursuit.append(normalized_mse(recons, acts_gpu).item())
    fvus_to_plot_grad_pursuit.append(fraction_variance_unexplained(acts_gpu, recons).item())

    x_gpu = acts_gpu
    y_gpu = acts.to(device) - recons
    predictions = x_gpu @ run_lstsq(x_gpu, y_gpu, device=device)[2].to(device)

    normalized_mse_with_preds_to_plot_grad_pursuit.append(normalized_mse(recons + predictions, acts_gpu).item())
    fvus_with_preds_to_plot_grad_pursuit.append(fraction_variance_unexplained(acts_gpu, recons + predictions).item())

with open(f"{BASE_DIR}/data/grad_pursuit_mses_{size}.pkl", "wb") as f:
    pickle.dump((normalized_mses_to_plot_grad_pursuit, normalized_mse_with_preds_to_plot_grad_pursuit, fvus_to_plot_grad_pursuit, fvus_with_preds_to_plot_grad_pursuit), f)



# %%

# ------------------- MAKE PLOTS -------------------

with open(f"{BASE_DIR}/data/grad_pursuit_mses_{size}.pkl", "rb") as f:
    normalized_mses_to_plot_grad_pursuit, normalized_mse_with_preds_to_plot_grad_pursuit, fvus_to_plot_grad_pursuit, fvus_with_preds_to_plot_grad_pursuit = pickle.load(f)

with open(f"{BASE_DIR}/data/sae_power_laws_{size}.pkl", "rb") as f:
    saes, normalized_mses, normalized_mses_with_predictions, fvus, fvus_with_predictions, empirical_l0s, sae_error_norm_r_squareds, sae_error_vec_r_squareds = pickle.load(f)

# %%

plot_type = "fvu"
# plot_type = "mse"

if plot_type == "mse":
    losses = normalized_mses
    losses_with_preds = normalized_mses_with_predictions
    losses_grad_pursuit = normalized_mses_to_plot_grad_pursuit
    losses_grad_pursuit_with_preds = normalized_mse_with_preds_to_plot_grad_pursuit
    ylabel = "Normalized MSE"
else:
    losses = fvus
    losses_with_preds = fvus_with_predictions
    losses_grad_pursuit = fvus_to_plot_grad_pursuit
    losses_grad_pursuit_with_preds = fvus_with_preds_to_plot_grad_pursuit
    ylabel = "FVU"

# %%

sae_widths = ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]

# Sort saes
correct_order_saes = []
correct_order_losses = []
correct_order_losses_with_preds = []
correct_order_empirical_l0s = []
correct_order_sae_error_norm_r_squareds = []
correct_order_sae_error_vec_r_squareds = []
for width in sae_widths:
    for i, sae in enumerate(saes):
        if sae[0] == width:
            correct_order_saes.append(sae)
            correct_order_losses.append(losses[i])
            correct_order_losses_with_preds.append(losses_with_preds[i])
            correct_order_empirical_l0s.append(empirical_l0s[i])
            correct_order_sae_error_norm_r_squareds.append(sae_error_norm_r_squareds[i])
            correct_order_sae_error_vec_r_squareds.append(sae_error_vec_r_squareds[i])
            break
saes = correct_order_saes
losses = correct_order_losses
losses_with_preds = correct_order_losses_with_preds
empirical_l0s = correct_order_empirical_l0s
sae_error_norm_r_squareds = correct_order_sae_error_norm_r_squareds
sae_error_vec_r_squareds = correct_order_sae_error_vec_r_squareds


# %%

widths = [2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]
all_widths = []
all_losses_with_preds = []
all_l0s = []

for width, sae, losses_with_preds_local in zip(widths, saes, losses_with_preds):
    for l0, loss in zip(sae[1], losses_with_preds_local):
        all_widths.append(width)
        all_losses_with_preds.append(loss)
        all_l0s.append(l0)

# Sample data (replace this with your actual data)
x_contour = np.array(all_widths)
y_contour = np.array(all_l0s)
z_contour = np.array(all_losses_with_preds)

xi = np.logspace(np.log10(x_contour.min()), np.log10(x_contour.max()), 100)
yi = np.logspace(np.log10(y_contour.min()), np.log10(y_contour.max()), 100)
XI, YI = np.meshgrid(xi, yi)

# Interpolate scattered data to a grid
ZI = griddata((x_contour, y_contour), z_contour, (XI, YI), method='linear')

# Create the contour plot
fig, ax = plt.subplots(1, 1, figsize=(2.75, 2))
contour = plt.contourf(XI, YI, ZI, levels=15, cmap='viridis')
cbar = plt.colorbar(contour)
cbar.ax.tick_params(labelsize=6) 
cbar.set_label(label=f"Nonlinear SAE {ylabel}",size=7, y=0.44)
ax.scatter(x_contour, y_contour, c='black', s=1, alpha=0.7)
ax.set_xlabel("SAE Width", fontsize=8)
ax.set_ylabel('SAE L0', fontsize=8)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=7)
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig("plots/sae_power_law_contour.pdf", bbox_inches='tight', pad_inches=0.02)
plt.show()
plt.close()


# %%

off_by_limit = 0.2

target_l0 = 60
print("Target L0:", target_l0)

losses_l0_60 = []
losses_l0_60_with_preds = []

for i, (width, l0s, losses_for_width, losses_with_preds_for_width) in enumerate(zip(sae_widths, empirical_l0s, losses, losses_with_preds)):
    l0 = get_l0_closest_to(l0s, target_l0)
    print(l0)
    if abs(l0 - target_l0) > target_l0 * off_by_limit:
        print(f"Bad!")

    index = l0s.index(l0)
    losses_l0_60.append(losses_for_width[index])
    losses_l0_60_with_preds.append(losses_with_preds_for_width[index])
    
# %%


if to_plot == "pursuit":
    losses_l0_60 = losses_grad_pursuit
    losses_l0_60_with_preds = losses_grad_pursuit_with_preds

torch.set_grad_enabled(True)

class PowerLawModel(nn.Module):
    def __init__(self, with_c=True):
        super(PowerLawModel, self).__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.m = nn.Parameter(torch.randn(1))
        self.c = nn.Parameter(torch.randn(1))
        self.with_c = with_c

    def forward(self, x):
        if not self.with_c:
            return torch.abs(self.a) * torch.pow(x, -torch.abs(self.m))
        return torch.abs(self.c) + torch.abs(self.a) * torch.pow(x, -torch.abs(self.m))

    def get_params(self):
        if not self.with_c:
            return torch.abs(self.a), -torch.abs(self.m)
        return torch.abs(self.c), torch.abs(self.a), -torch.abs(self.m)

x = torch.tensor(widths, dtype=torch.float32)
y = torch.tensor(losses_l0_60, dtype=torch.float32)

model = PowerLawModel(with_c=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001) 

num_epochs = 50000
for epoch in range(num_epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# %%

fig, ax = plt.subplots(1, 1, figsize=(2.75, 2))

cmap = get_cmap('viridis')
color_step = 1.0 / 7
colors = [cmap(1 * color_step), cmap(5 * color_step)]

ax.plot(widths, losses_l0_60, "o", label="SAE Reconstruction", color=colors[0], markersize=3)

x = np.logspace(np.log10(min(x)), np.log10(max(x)), 100, base=10)
y = model(torch.tensor(x, dtype=torch.float32)).detach().numpy()
c, a, m = model.get_params()
c, a, m = c.item(), a.item(), m.item()
ax.plot(x, y, label=f'${{{ylabel}}} \\approx {{{c:.3f}}} + {{{a:.3f}}} W^{{{m:.3f}}}$', color=colors[0], linestyle="--", linewidth=1)


# Plot horizontal line at average 
ax.plot(widths, losses_l0_60_with_preds, "o", label="Unexplained Noise", color=colors[1], markersize=3)
average = np.mean(losses_l0_60_with_preds)
ax.axhline(average, color=colors[1], linestyle='--', label=f"${{{ylabel}}} \\approx {{{average:.3f}}}$", linewidth=1)

ax.set_xscale("log")
ax.set_yscale("log")
# Set there to be 5 ticks on y axis
# ticks = np.logspace(-1.2, -0.6, 8)
# ax.set_yticks(ticks, [f"{t:.2f}" for t in ticks])
# ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(bbox_to_anchor=(0.01, 0.62), loc='upper left', fontsize=7)
ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=7)

# plt.title(f"L0 $\\approx$ 60")
plt.xlabel("SAE Width", fontsize=8)
plt.ylabel(ylabel, fontsize=8)
plt.savefig(f"plots/sae_power_law_fit_{target_l0}.pdf", bbox_inches='tight', pad_inches=0.02)

# %%

widths = [2**14, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]
all_zs = [sae_error_norm_r_squareds, sae_error_vec_r_squareds]
labels = ["norm", "vec"]

def create_contour_data(widths, saes, z_data):
    all_widths = []
    all_z = []
    all_l0s = []
    for width, sae, z_local in zip(widths, saes, z_data):
        for l0, z in zip(sae[1], z_local):
            all_widths.append(width)
            all_z.append(z)
            all_l0s.append(l0)
    return np.array(all_widths), np.array(all_l0s), np.array(all_z)

for i, (z_data, label) in enumerate(zip(all_zs, labels)):

    fig, ax = plt.subplots(1, 1, figsize=(2.75, 2.5), sharey=True)

    x_contour, y_contour, z_contour = create_contour_data(widths, saes, z_data)
    
    # Expand the range slightly for interpolation
    x_min, x_max = x_contour.min() / 1.15, x_contour.max() * 1.15
    y_min, y_max = y_contour.min() / 1.15, y_contour.max() * 1.15
    
    xi = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    yi = np.logspace(np.log10(y_min), np.log10(y_max), 100)
    XI, YI = np.meshgrid(xi, yi)
    
    # Interpolate scattered data to a grid
    ZI = griddata((x_contour, y_contour), z_contour, (XI, YI), method='linear')
    
    # Create the contour plot
    contour = ax.contourf(XI, YI, ZI, levels=15, cmap='viridis', extend='both')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.tick_params(labelsize=6)
    
    ax.scatter(x_contour, y_contour, c='black', s=1, alpha=0.7)
    ax.set_xlabel("SAE Width", fontsize=8)
    ax.set_ylabel('SAE L0', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=7)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_title(label, fontsize=10)
    
    # Set expanded limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # plt.tight_layout()
    plt.savefig(f"plots/sae_power_law_contours_{label}.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.show()
    plt.close()

# %%

if to_plot == "both":
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.5))
else:
    fig, ax = plt.subplots(1, 1, figsize=(2.75, 2))

cmap = get_cmap('viridis')
color_step = 1.0 / 7
colors = [cmap(1 * color_step), cmap(3 * color_step), cmap(5 * color_step), cmap(7 * color_step)]

extended_x = np.logspace(np.log10(min(x) / 1.3), np.log10(max(x) * 10), 100, base=10)
extended_y = model(torch.tensor(extended_x, dtype=torch.float32)).detach().numpy()

# Split the curve into solid and dotted parts
last_x = widths[-1]
x_solid = extended_x[extended_x <= last_x]
x_dotted = extended_x[extended_x > last_x]
y_solid = extended_y[:len(x_solid)]
y_dotted = extended_y[len(x_solid):]

plot_log = False

if plot_log:
    if plot_type == "fvu":
        ax.set_ylim(0.055, 0.37)
    else:
        ax.set_ylim(0.055, 0.25)
    ax.set_xlim(min(extended_x), max(extended_x))
    ax.set_xscale("log")
    ax.set_yscale("log")
else:
    if plot_type == "fvu":
        ax.set_ylim(0, 0.37)
    else:
        ax.set_ylim(0, 0.24)
    ax.set_xlim(min(extended_x), max(extended_x))
    ax.set_xscale("log")
# ax.set_yscale("log")

# Get the full x-range of the plot
x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

ax.tick_params(axis='both', which='major', labelsize=7)
ax.tick_params(axis='both', which='minor', labelsize=7)

# Turn off the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax.margins(x=0)






# SAE Reconstruction
ax.plot(widths, losses_l0_60, "o", label="SAE Reconstruction", color=colors[0], markersize=3)
ax.plot(x_solid, y_solid, color=colors[0], linestyle="-", linewidth=1)
ax.plot(x_dotted, y_dotted, color=colors[0], linestyle=":", linewidth=1.3)

# Decoder/Sparsity Error, normal
ax.plot([x_min, last_x], [average, average], color=colors[1], linestyle='-', linewidth=1)
ax.plot([last_x, x_max], [average, average], color=colors[1], linestyle=':', linewidth=1.3)
ax.plot(widths, losses_l0_60_with_preds, "o", color=colors[1], markersize=3)

# SAE Reconstruction limit line (c)
ax.plot([x_min, x_max], [c, c], color=colors[0], linestyle=':', linewidth=1.3)

# Decoder/Sparsity Error, pursuit
if to_plot == "both":
    ax.plot(widths, losses_grad_pursuit_with_preds, "o", color=colors[2], markersize=3)

# Extend x for fill_between
x_extended = np.logspace(np.log10(x_min), np.log10(x_max), 1000)
y_extended = model(torch.tensor(x_extended, dtype=torch.float32)).detach().numpy()

# Fill between the curves and axes
ax.fill_between(x_extended, y_extended, c, alpha=0.3, color=colors[0])
ax.fill_between(x_extended, c, average, alpha=0.3, color=colors[1])
if to_plot == "both":
    def piecewise_encoder_boundary(x):
        # Find the last x value where we have actual data
        last_data_x = widths[-1]
        last_data_y = losses_grad_pursuit_with_preds[-1]
        
        # Create a piecewise function
        if x <= last_data_x:
            # Interpolate between known points for x values within data range
            return np.interp(x, widths, losses_grad_pursuit_with_preds)
        else:
            # Use the last known y value for x values beyond data range
            return last_data_y

    encoder_boundary = np.vectorize(piecewise_encoder_boundary)(x_extended)

    # Plot the piecewise encoder boundary
    ax.plot(x_extended[x_extended <= widths[-1]], encoder_boundary[x_extended <= widths[-1]], 
            color=colors[2], linewidth=1)
    ax.plot(x_extended[x_extended > widths[-1]], encoder_boundary[x_extended > widths[-1]], 
            color=colors[2], linestyle=':', linewidth=1.3)

    # Fill between the curves
    ax.fill_between(x_extended, average, encoder_boundary, alpha=0.3, color=colors[2])
    ax.fill_between(x_extended, encoder_boundary, y_min, alpha=0.3, color=colors[3])
else:
    ax.fill_between(x_extended, average, y_min, alpha=0.3, color=colors[3])

# Add labels to each section
if plot_log:
    average_func = lambda x, y: np.sqrt(x * y)
else:
    average_func = lambda x, y: (x + y) / 2 - 0.003

main_fontsize = 10 if to_plot == "both" else 7
ax.text(20_000, average_func(c, losses_l0_60[3]), 'Absent Features', fontsize=main_fontsize, ha='left', va='center')
ax.text(20_000, average_func(c, average), 'Linear Error', fontsize=main_fontsize, ha='left', va='center')
ax.text(20_000, average_func(y_min, average) - 0.01, 'Nonlinear Error', fontsize=main_fontsize, ha='left', va='center')


# Legend
custom_lines = [Line2D([0], [0], color=colors[0], marker='o', linestyle='None', markersize=3),
                Line2D([0], [0], color=colors[1], marker='o', linestyle='None', markersize=3),
                Line2D([0], [0], color=colors[2], marker='o', linestyle='None', markersize=3)]


if to_plot == "pursuit":
    ax.legend(custom_lines, 
            [f'Pursuit SAE Reconstruction, ${{{ylabel}}} \\approx {{{c:.3f}}} + {{{a:.3f}}} W^{{{m:.3f}}}$',
            f'Pursuit SAE Reconstruction + Error Prediction, ${{{ylabel}}} \\approx {{{average:.3f}}}$'],
            fontsize=6, ncol=1, bbox_to_anchor=(-0.2, 1), loc='lower left')
else:
    ax.legend(custom_lines, 
            [f'SAE Reconstruction, ${{{ylabel}}} \\approx {{{c:.3f}}} + {{{a:.3f}}} W^{{{m:.3f}}}$',
            f'SAE Reconstruction + Error Prediction, ${{{ylabel}}} \\approx {{{average:.3f}}}$',
            "SAE Pursuit Reconstruction + Error Prediction"],
            loc='upper right', fontsize=7, ncol=1)

if to_plot == "both":
    # Add zoomed inset
    axins = inset_axes(ax, width="34%", height="20%", loc='lower right', 
                    bbox_transform=ax.transAxes)

    # Plot the same data in the inset
    axins.plot(widths, losses_l0_60, "o", color=colors[0], markersize=3)
    axins.plot(x_solid, y_solid, color=colors[0], linestyle="-", linewidth=1)
    axins.plot(x_dotted, y_dotted, color=colors[0], linestyle=":", linewidth=1.3)

    axins.plot([x_min, last_x], [average, average], color=colors[1], linestyle='-', linewidth=1)
    axins.plot([last_x, x_max], [average, average], color=colors[1], linestyle=':', linewidth=1.3)
    # axins.plot(widths, losses_l0_60_with_preds, "o", color=colors[1], markersize=3)

    if to_plot == "both":
        # axins.plot(widths, losses_grad_pursuit_with_preds, "o", color=colors[2], markersize=3)
        axins.plot(x_extended[x_extended <= widths[-1]], encoder_boundary[x_extended <= widths[-1]], 
                color=colors[2], linewidth=1)
        axins.plot(x_extended[x_extended > widths[-1]], encoder_boundary[x_extended > widths[-1]], 
                color=colors[2], linestyle=':', linewidth=1.3)

        axins.fill_between(x_extended, y_extended, c, alpha=0.3, color=colors[0])
        axins.fill_between(x_extended, c, average, alpha=0.3, color=colors[1])
        axins.fill_between(x_extended, average, encoder_boundary, alpha=0.3, color=colors[2])
        axins.fill_between(x_extended, encoder_boundary, y_min, alpha=0.3, color=colors[3])

    # Set the limits for the zoomed area
    x1, x2 = 200_000, 1e7  # Adjust these values to zoom into the desired x-range
    y1, y2 = average - 0.015, average + 0.004  # Adjust these values to zoom into the desired y-range
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.set_xscale('log')

    # Remove axes ticks for inset
    axins.set_xticks([])
    axins.set_yticks([])


    # Draw a box around the zoomed region
    # mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")


    if to_plot == "both":
        axins.text(0.65, 0.3, 'Encoder Error', transform=axins.transAxes,
                ha='center', va='bottom', fontsize=10)

    con1 = ConnectionPatch(
        xyA=(x1, y1), xyB=(0, 1), coordsA=ax.transData, coordsB=axins.transAxes,
        axesA=ax, axesB=axins, color="0.5", linestyle="-"
    )
    con2 = ConnectionPatch(
        xyA=(x2, y1), xyB=(1, 1), coordsA=ax.transData, coordsB=axins.transAxes,
        axesA=ax, axesB=axins, color="0.5", linestyle="-"
    )

    # Add the connector lines to the main axes
    ax.add_artist(con1)
    ax.add_artist(con2)

    # Add a rectangle around the zoomed region in the main plot
    rect = plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, fill=False, ec="0.5", linestyle="-", transform=ax.transData
    )
    ax.add_patch(rect)


if to_plot == "both":
    fig.supxlabel("SAE Width", fontsize=8, y=-0.01)
    fig.supylabel("Fraction Variance Unexplained (FVU)" if ylabel == "FVU" else ylabel,  fontsize=8, x=0.04, y=0.47)
else:
    fig.supxlabel("SAE Width", fontsize=8, y=-0.06)
    fig.supylabel(ylabel, fontsize=8, x=-0.01)
if to_plot == "both":
    plt.savefig("plots/scaling_filled_with_zoom_combined.pdf", bbox_inches='tight', pad_inches=0.02)
else:
    plt.savefig(f"plots/scaling_filled_{to_plot}.pdf", bbox_inches='tight', pad_inches=0.02)
plt.show()
# %%
