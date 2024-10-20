# %%
from utils import get_sae_info, get_gemma_sae_params_for_layer, get_l0_closest_to, get_sae_info_by_params
from sae_lens import SAE
import torch
from tqdm import tqdm
from sklearn.linear_model import OrthogonalMatchingPursuit
import argparse
import einops
from utils import BASE_DIR
import os

os.makedirs(f"{BASE_DIR}/data", exist_ok=True)

torch.set_grad_enabled(False)

# %%

size = "9b"

argparser = argparse.ArgumentParser()
argparser.add_argument("--layer", type=int, default=20)
argparser.add_argument("--width", type=str, default="131k")
argparser.add_argument("--base_l0", type=int, default=114)
argparser.add_argument("--device", type=str, default="cuda:0")
argparser.add_argument("--save_incremental", action="store_true")
argparser.add_argument("--batch_size", type=int, default=256)

args = argparser.parse_args()
l0 = args.base_l0
width = args.width
device = args.device
save_incremental = args.save_incremental
batch_size = args.batch_size
layer = args.layer

sae_name = f"layer_{layer}/width_{width}/average_l0_{l0}"
dictionary_sae = SAE.from_pretrained(
    release=f"gemma-scope-{size}-pt-res",
    sae_id=sae_name,
    device="cpu",
)[0] 

sae_info = get_sae_info(layer=layer, sae_name=sae_name, model="gemma_2_9b")

# %%


def grad_pursuit_update_step(signals, weights, dictionary):
    """
    Performs a single gradient pursuit update step for a batch of signals.

    Parameters:
    - signal: Tensor of shape (b, d,)
    - weights: Tensor of shape (b, n,)
    - dictionary: Tensor of shape (n, d)

    Returns:
    - Updated weights tensor of shape (b, n,)
    """
    # Compute the residual
    reconstructed_signal = torch.matmul(weights, dictionary)  # Shape: (b, d,)
    residual = signals - reconstructed_signal  # Shape: (b, d,)

    # Get a mask for selected features
    selected_features = (weights != 0)  # Shape: (b, n,)

    # Compute inner products between dictionary atoms and residual
    inner_products = torch.matmul(residual, dictionary.T)  # Shape: (b, n,)

    # Select the index with the largest inner product
    idx = torch.argmax(inner_products, dim=-1)  # Shape: (b,)
    selected_features[torch.arange(selected_features.shape[0]), idx] = True

    # Compute the gradient for the weights
    grad = inner_products * selected_features.float()  # Shape: (b, n,)

    # Compute the optimal step size
    c = torch.matmul(grad, dictionary)  # Shape: (b, d)
    numerator = einops.einsum(c, residual, "b d, b d -> b")  # Shape: (b,)
    denominator = einops.einsum(c, c, "b d, b d -> b")  # Shape: (b,)
    step_size = numerator / denominator # Shape: (b,)

    # Update the weights
    weights = weights + einops.einsum(grad, step_size, "b n, b -> b n")  # Shape: (b, n,)

    # Clip the weights to be non-negative
    weights = torch.clamp(weights, min=0)

    return weights

def grad_pursuit(signals, dictionary, target_l0s):
    """
    Performs gradient pursuit to approximate the signal using a sparse combination of dictionary atoms.

    Parameters:
    - signals: Tensor of shape (b, d)
    - dictionary: Tensor of shape (n, d)
    - target_l0s: Tensor of shape (b), the target sparsity levels (number of non-zero weights)

    Returns:
    - Weights tensor of shape (b, n) representing the sparse representations of the signals
    """
    n = dictionary.shape[0]
    max_l0 = target_l0s.max().item()
    target_l0s = target_l0s.to(signals.device)
    weights = torch.zeros(len(signals), n, dtype=signals.dtype, device=signals.device)
    final_weights = torch.zeros_like(weights, device=signals.device)
    for current_l0 in range(max_l0):
        weights = grad_pursuit_update_step(signals, weights, dictionary)
        final_weights = torch.where(target_l0s.unsqueeze(1) > current_l0, weights, final_weights)
    return final_weights


def run_grad_pursuit_and_save_recons(dictionary_sae, sae_info, batch_size=1):

    original_signals = sae_info.acts_flattened
    l0s = sae_info.l0s_flattened

    bias = dictionary_sae.b_dec.detach()
    unbiased_signals = original_signals - bias

    dictionary = dictionary_sae.W_dec.data.detach().to(device)

    dict_sae_name_no_slashes = sae_name.replace("/", "_")
    original_sae_name_no_slashes = sae_info.sae_name.replace("/", "_")

    total_mse_grad_pursuit = 0
    pursuit_reconstructions = []
    bar = tqdm(range(0, len(original_signals), batch_size))
    for i in bar:
        original_signals_batch = original_signals[i:i + batch_size].to(device)
        unbiased_signals_batch = unbiased_signals[i:i + batch_size].to(device)

        target_l0s = l0s[i:i + batch_size]

        results = grad_pursuit(unbiased_signals_batch, dictionary, target_l0s)
        reconstructions = results @ dictionary + bias.to(device)
        total_mse_grad_pursuit += ((original_signals_batch - reconstructions) ** 2).sum()
        pursuit_reconstructions.append(reconstructions.to("cpu"))

        bar.set_description(f"{total_mse_grad_pursuit / (i + 1) / batch_size:.2f}")

        if save_incremental and i % 100 == 0:
            torch.save(torch.concat(pursuit_reconstructions), f"{BASE_DIR}/data/grad_pursuit_reconstructions_{original_sae_name_no_slashes}-{dict_sae_name_no_slashes}_incremental.pt")

    pursuit_reconstructions = torch.concat(pursuit_reconstructions)
    torch.save(pursuit_reconstructions, f"{BASE_DIR}/data/grad_pursuit_reconstructions_{original_sae_name_no_slashes}-{dict_sae_name_no_slashes}.pt")

run_grad_pursuit_and_save_recons(dictionary_sae, sae_info, batch_size=batch_size)
