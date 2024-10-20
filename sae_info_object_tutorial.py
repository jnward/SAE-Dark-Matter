# %%

from utils import run_lstsq, get_sae_info, get_gemma_sae_params_for_layer, get_l0_closest_to, get_sae_info_by_params

# %%

# Load an SAEInfoObject and access some fields
# The SAEInfoObject contains flattened results of running 300k tokens through the model, 
# skipping the first 200 tokens of each context (by default, this can be changed in the SAEInfoObject constructor)

example_sae_info = get_sae_info(layer="20", sae_name="layer_20/width_131k/average_l0_12", model="gemma_2_9b")

model_losses_flattened = example_sae_info.model_loss_flattened
sae_error_norms_flattened = example_sae_info.sae_error_norms_flattened
l0s_flattened = example_sae_info.l0s_flattened
act_norm_flattened = example_sae_info.act_norm_flattened
tokens_flattened = example_sae_info.tokens_flattened
ones = example_sae_info.ones
max_token = example_sae_info.max_token
acts_flattened = example_sae_info.acts_flattened
sae_error_vecs_flattened = example_sae_info.sae_error_vecs_flattened
active_sae_features = example_sae_info.active_sae_features
context_start_threshold = example_sae_info.context_start_threshold
context_end_threshold = example_sae_info.context_end_threshold
sae_name = example_sae_info.sae_name

# %%

# Get list of all SAEs that you can load into SAE infos
sae_params = get_gemma_sae_params_for_layer(layer=20, model="gemma_2_9b")

print(sae_params)

# %%

# Get SAE info for one of the params
width = "16k"
l0s = sae_params[width]
target_l0 = 130
actual_l0 = get_l0_closest_to(l0s, target_l0)
print(width, actual_l0)
sae_info = get_sae_info_by_params(layer="20", sae_width=width, sae_l0=actual_l0, model="gemma_2_9b")

# %%

# Run a least squares regression to predict sae errors from activations
device = "cuda:0"
x = acts_flattened.to(device)
y = sae_error_vecs_flattened.to(device)
residuals, r_squared, solution = run_lstsq(x, y, device=device)
print(residuals.shape, r_squared, solution.shape)
# %%

# Load an MLP SAE info
sae_params_mlps = get_gemma_sae_params_for_layer(layer=20, model="gemma_2_9b", layer_type="mlp")
width = "16k"
target_l0 = 100
actual_l0 = get_l0_closest_to(sae_params_mlps[width], target_l0)
print(width, actual_l0)
sae_info_mlp = get_sae_info_by_params(layer="20", sae_width=width, sae_l0=actual_l0, model="gemma_2_9b", layer_type="mlp")
print(sae_info_mlp.acts_flattened.shape)

# %%

# Load an attention SAE info
# Some of the extra fields like sae_error_norms might not quite have the right shape, but the main ones shown below should be okay
sae_params_att = get_gemma_sae_params_for_layer(layer=20, model="gemma_2_9b", layer_type="att")
width = "16k"
target_l0 = 100
actual_l0 = get_l0_closest_to(sae_params_att[width], target_l0)
print(width, actual_l0)
sae_info_att = get_sae_info_by_params(layer="20", sae_width=width, sae_l0=actual_l0, model="gemma_2_9b", layer_type="att")
print(sae_info_att.acts_flattened.shape)
print(sae_info_att.sae_error_vecs_flattened.shape)
