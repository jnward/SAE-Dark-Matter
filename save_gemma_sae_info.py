# %%

import transformer_lens
from sae_lens import SAE
from datasets import load_dataset
import torch
from tqdm import tqdm
import os
import argparse
import numpy as np
from utils import BASE_DIR

torch.set_grad_enabled(False)


# %%

argparser = argparse.ArgumentParser()
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--size", type=str, default="2b")
argparser.add_argument("--layer", type=int, default=12)
argparser.add_argument("--sae_width", type=str, default="16k")
argparser.add_argument("--sae_l0", type=int, required=True)
argparser.add_argument("--layer_type", choices=["res", "mlp", "att"], default="res")

args = argparser.parse_args()

size = args.size
layer = args.layer
device = args.device
sae_width = args.sae_width
sae_l0 = args.sae_l0
layer_type = args.layer_type

# %%

sae_name = f"layer_{layer}/width_{sae_width}/average_l0_{sae_l0}"
if layer_type != "res":
    save_dir_base = f"{BASE_DIR}/gemma_{size}_sae_scaling_{args.layer_type}"
else:
    save_dir_base = f"{BASE_DIR}/gemma_{size}_sae_scaling"
save_dir = f"{save_dir_base}/{sae_name}"
os.makedirs(save_dir, exist_ok=True)

# %%

# acts = torch.load(f"{save_dir_base}/acts_layer_{layer}.pt")
if layer_type != "att":
    if size == "2b":
        acts_shape = [300, 1024, 2304]
    else:
        acts_shape = [300, 1024, 3584]
else:
    if size == "2b":
        raise NotImplementedError()
    else:
        acts_shape = [300, 1024, 16, 256]
acts = torch.from_file(f"{save_dir_base}/acts_layer_{layer}.npy", shared=False, size=np.prod(acts_shape)).view(*acts_shape)

# %%
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=f"gemma-scope-{size}-pt-{layer_type}",
    sae_id=sae_name,
    device=device,
)

sae = sae.to(torch.bfloat16)

hook_name = sae.cfg.hook_name
print(hook_name)

# %%

batch_size = 1
if layer_type != "att":
    num_contexts, ctx_len, dim = acts.shape
else:
    num_contexts, ctx_len, num_heads, head_dim = acts.shape
    
total_tokens = num_contexts * ctx_len
print(f"Total tokens: {total_tokens / 1e6:.2f}M")

# %%


all_sae_l0s = []
all_sae_errors = []
all_feature_act_norms = []
all_sae_error_vecs = []
all_sae_features_acts = []


def save_so_far():
    all_sae_l0s_cat = torch.cat(all_sae_l0s, dim=0).float()
    torch.save(all_sae_l0s_cat, f"{save_dir}/sae_l0s_layer_{layer}.pt")
    all_sae_errors_cat = torch.cat(all_sae_errors, dim=0).float()
    torch.save(all_sae_errors_cat, f"{save_dir}/sae_errors_layer_{layer}.pt")
    all_sae_error_vecs_cat = torch.cat(all_sae_error_vecs, dim=0).float()
    all_sae_error_vecs_cat.numpy().tofile(f"{save_dir}/sae_error_vecs_layer_{layer}.npy")
    all_feature_act_norms_cat = torch.cat(all_feature_act_norms, dim=0).float()
    torch.save(
        all_feature_act_norms_cat, f"{save_dir}/feature_act_norms_layer_{layer}.pt"
    )

    torch.save(all_sae_features_acts, f"{save_dir}/active_sae_features_layer_{layer}.pt")


# %%


bar = tqdm(range(0, num_contexts, batch_size))
for i in bar:
    acts_batch_cpu = acts[i : i + batch_size]
    acts_batch = acts_batch_cpu.clone().to(device)

    feature_acts = sae.encode(acts_batch)
    reconstructions = sae.decode(feature_acts)
    feature_acts_cpu = feature_acts.to("cpu")
    l0s = (feature_acts_cpu > 0).sum(dim=-1)
    feature_acts_norms = feature_acts_cpu.norm(dim=-1)

    for j in range(batch_size):
        feature_acts_j = feature_acts_cpu[j]
        nonzero_feature_indices_j = torch.nonzero(feature_acts_j, as_tuple=True)
        nonzero_feature_values_j = feature_acts_j[nonzero_feature_indices_j]
        res = [nonzero_feature_indices_j[0], nonzero_feature_indices_j[1], nonzero_feature_values_j]
        all_sae_features_acts.append(res)

    # Get reconstruction error
    reconstructions_cpu = reconstructions.to("cpu")
    reconstruction_errors = reconstructions_cpu - acts_batch_cpu
    reconstruction_error_norms = reconstruction_errors.square().sum(dim=-1)

    all_sae_l0s.append(l0s.to("cpu"))
    all_sae_errors.append(reconstruction_error_norms.to("cpu"))
    all_sae_error_vecs.append(reconstruction_errors.to("cpu"))
    all_feature_act_norms.append(feature_acts_norms.to("cpu"))

    bar.set_description(
        f"Num tokens: {len(all_sae_errors) * batch_size * ctx_len / 1e6:.2f}M"
    )

save_so_far()

# %%
