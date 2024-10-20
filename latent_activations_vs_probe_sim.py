# %%
from utils import get_sae_info, run_lstsq
import torch
from sae_lens import SAE
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

# %%

sae_info = get_sae_info(layer="20", sae_name="layer_20/width_131k/average_l0_62", model="gemma_2_9b")

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=f"gemma-scope-9b-pt-res",
    sae_id=sae_info.sae_name,
    device="cpu",
)

# %%

data = (sae_info.reconstruction_vecs_flattened - sae.b_dec)
resids, r_squared, sol = run_lstsq(data, data.norm(dim=-1)**2)
# %%
average_feature_values_when_active = sae_info.get_feature_freqs()[1]
predictions = []
actuals = []
for i in range(len(average_feature_values_when_active)):
    predictions.append(sae.W_dec[i] @ sol)
    actuals.append(average_feature_values_when_active[i])

# Filter out nans 
nan_indices = torch.isnan(torch.tensor(actuals))
predictions = [predictions[i] for i in range(len(predictions)) if not nan_indices[i]]
actuals = [actuals[i] for i in range(len(actuals)) if not nan_indices[i]]

# Sort by magnitude of actual
sorted_indices = torch.argsort(torch.tensor(actuals), descending=True)
predictions = [predictions[i] for i in sorted_indices]
actuals = [actuals[i] for i in sorted_indices]

# %%
smoothed = torch.nn.functional.avg_pool1d(torch.tensor(predictions).unsqueeze(0), 100, stride=1, padding=5).squeeze()
limit = 10000
fig, ax = plt.subplots(figsize=(2.75, 2))
ax.plot(predictions[:limit], label="$(a^*)^T \cdot feature$")
ax.plot(smoothed[:limit], label="Running window average of $(a^*)^T \cdot feature$")
ax.plot(actuals[:limit], label="Average feature value when active")
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.legend(fontsize=7, ncol=1, bbox_to_anchor=(0.5, 1.24), loc="center")
plt.ylim(-200, 300)
plt.xlabel("SAE Latent Index\nSorted by Average Active Value", fontsize=7)
plt.savefig("plots/feature_activation_predictions.pdf", bbox_inches="tight")