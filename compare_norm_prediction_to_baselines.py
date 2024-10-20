# %%

from utils import get_sae_info_by_params, get_gemma_sae_params_for_layer, get_l0_closest_to, run_lstsq, calculate_r_squared
import torch
import matplotlib.pyplot as plt

target_l0 = 60
device = "cpu"


def calculate_per_token_stats(max_token, tokens, stats, offset=0):

    token_sums = torch.zeros(max_token + 1, device=device)

    token_counts = torch.bincount(tokens, minlength=max_token + 1).to(device)

    token_sums.index_add_(0, tokens.to(device), stats.to(device))

    # Calculate average residuals
    token_sums = torch.where(
        token_counts > 0, token_sums / token_counts, torch.zeros_like(token_sums)
    )

    return token_counts, token_sums


act_r_squareds = []
token_r_squareds = []
l0_r_squareds = []
activation_norm_r_squareds = []
model_loss_r_squareds = []

layers = range(40)
for layer in layers:
    params = get_gemma_sae_params_for_layer(layer=layer, model="gemma_2_9b")
    l0s = params["131k"]
    closest_l0 = get_l0_closest_to(l0s, target_l0)
    sae_info = get_sae_info_by_params(layer=layer, sae_width="131k", sae_l0=closest_l0, model="gemma_2_9b")

    y = sae_info.sae_error_norms_flattened

    x = sae_info.acts_flattened
    ones = torch.ones(x.shape[0], 1)
    x_with_ones = torch.cat([x, ones], dim=1)
    act_r_squareds.append(run_lstsq(x_with_ones, y)[1])

    # Just predict average token SAE error norm
    token_counts, token_sums = calculate_per_token_stats(
        sae_info.max_token,
        sae_info.tokens_flattened,
        y
    )
    token_predictions = token_sums[sae_info.tokens_flattened]
    token_r_squareds.append(calculate_r_squared(y, y - token_predictions))

    x = sae_info.model_loss_flattened[:, None]
    x_with_ones = torch.cat([x, ones], dim=1)
    model_loss_r_squareds.append(run_lstsq(x_with_ones, y)[1])

    x = sae_info.l0s_flattened[:, None]
    x_with_ones = torch.cat([x, ones], dim=1)
    l0_r_squareds.append(run_lstsq(x_with_ones, y)[1])

    x = sae_info.acts_flattened.norm(dim=-1)[:, None]
    x_with_ones = torch.cat([x, ones], dim=1)
    activation_norm_r_squareds.append(run_lstsq(x_with_ones, y)[1])

    print(act_r_squareds[-1], token_r_squareds[-1], model_loss_r_squareds[-1], l0_r_squareds[-1], activation_norm_r_squareds[-1])


# %%

fig, ax = plt.subplots(figsize=(2.75, 2))
ax.plot(layers, act_r_squareds, label="Acts")
ax.plot(layers, token_r_squareds, label="Tokens")
ax.plot(layers, model_loss_r_squareds, label="Model Loss")
ax.plot(layers, l0_r_squareds, label="L0")
ax.plot(layers, activation_norm_r_squareds, label="Act Norm")
plt.xlabel("Layer", fontsize=7)
plt.ylabel("R^2", fontsize=7)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.legend(fontsize=7, ncol=3, bbox_to_anchor=(0.44, 1.2), loc="center")
plt.savefig("plots/comparing_norm_prediction_to_baselines.pdf", bbox_inches="tight")


