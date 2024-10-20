# %%

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from utils import run_lstsq, get_sae_info, get_gemma_sae_params_for_layer, get_l0_closest_to, get_sae_info_by_params
import torch
import torch
from functools import partial
from tqdm import tqdm
import gc
import pickle
from utils import BASE_DIR

torch.set_grad_enabled(False)

# Calculate the CE when removing each component
# 1. Learn a LR model for our target module (e.g. mlp, attn, resid)
# 2. Run new datapoints through, sepearting out our three components: SAE_rec, linear_error, noise
# 3. Replace each component w/ [baseline such as zeros or mean from (1)], and run through the rest of the model

# %%

import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--size", type=str, default="9b")
argument_parser.add_argument("--device", type=str, default="cuda:1")
argument_parser.add_argument("--loop_setting", type=str, default="width", choices=["width", "l0"])
argument_parser.add_argument("--model", type=str, default="gemma_2_9b")
argument_parser.add_argument("--layer", type=int, default=20)
argument_parser.add_argument("--layer_type", type=str, default="res", choices=["res", "att", "mlp"])

args = argument_parser.parse_args()
size = args.size
device = args.device
loop_setting = args.loop_setting
gemma_model = args.model
layer_type = "res"
layers = [args.layer]

# %%
size = "9b"
device = "cuda:0"
loop_setting = "width"
# loop_setting = "l0"
gemma_model = "gemma_2_9b"
layer_type = "res"
layers = [20]

# Dataset args
ctx_len = 1024
# ctx_len = 512
batch_size = 1
num_contexts = 300
dataset_name = "monology/pile-uncopyrighted"

# %%

model = HookedTransformer.from_pretrained_no_processing(
# model = HookedTransformer.from_pretrained(
    f"google/gemma-2-{size}",
    center_writing_weights=False,
    center_unembed=False,
    device=device,
    torch_dtype=torch.float16,
    # torch_dtype=torch.float32,
)

# %%



def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()


def tokenized_batch(data):
    batch = []
    while len(batch) < batch_size:
        next_text = next(data)
        tokenized = model.tokenizer(
            next_text,
            return_tensors="pt",
            max_length=ctx_len,
            padding=False,
            truncation=True,
        )
        if tokenized["input_ids"].shape[1] == ctx_len:
            batch.append(tokenized)
    return torch.cat([x["input_ids"] for x in batch], dim=0)



def calculate_r_squared(target_variable, residuals):
    total_sum_of_squares = torch.sum(
        (target_variable - torch.mean(target_variable, dim=0)) ** 2, dim=0
    )
    residual_sum_of_squares = torch.sum(residuals**2, dim=0)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    # If there are multiple targets, take the mean R^2
    # This is a bit of a hack, should probably find the variance explained in a more principled way
    return r_squared.mean().item()

from functools import partial
def inject_sae_baseline(x, hook, sae_vecs, linear_solution, setting, baseline=None):
    first_200_tokens = x[:, :200, :]

    x_hat = x.clone()
    x_hat[:, 200:-1, :] = sae_vecs.to(x.dtype)
    # x = x_hat + error
    sae_error = x - x_hat
    error_hat = x @ linear_solution

    noise = (sae_error - error_hat)
    # print(torch.allclose(noise + error_hat + x_hat, x, atol=1e-3))
    # if assertion error hits, give the difference
    if not torch.allclose((noise + error_hat + x_hat)[:, 200:, :], x[:, 200:, :], atol=1e-1):
        print(f"mean of x: {x.mean()}")
        diff = (noise + error_hat + x_hat -  x)[:, 200:, :]
        print(diff.norm())
        print(diff.max())
        print(torch.norm(diff))
        raise ValueError("Assertion error")

    # print("R2: ", calculate_r_squared(sae_error[0], error_hat[0]))
    def project_and_scale(a, b):
        # Project a onto b
        projection = (torch.dot(a.view(-1), b.view(-1)) / torch.dot(b.view(-1), b.view(-1))) * b
        # Calculate the norm of the projection
        norm_projection = torch.norm(projection)
        # Scale b to have the same norm as the projection
        scaled_b = norm_projection * b / torch.norm(b)
        return scaled_b

    if setting == "original":
        return x
    elif setting == "sae":
        result = x_hat
    elif setting == "sae_and_linear":
        # print(f"Norm of error_hat:", error_hat.norm().item())
        if baseline is not None:
            result = x_hat + error_hat + baseline
        else:
            result = x_hat + error_hat
    elif setting == "sae_and_noise":
        # print(f"Norm of noise:", noise.norm().item())
        if baseline is not None:
            result = x_hat + noise + baseline
        else:
            result = x_hat + noise
    # elif setting == "sae_and_error_hat_projection":
    #     scaled_error = project_and_scale(-error_hat, sae_error)
    #     # print(f"Norm of scaled error", scaled_error.norm().item())
    #     if baseline is not None:
    #         result = x_hat + scaled_error + baseline
    #     else:
    #         result = x_hat + scaled_error
    # elif setting == "sae_and_noise_projection":
    #     scaled_noise = project_and_scale(noise, sae_error)
    #     # print(f"Norm of scaled_noise:", scaled_noise.norm().item())
    #     if baseline is not None:
    #         result = x_hat + scaled_noise + baseline
    #     else:
    #         result = x_hat + scaled_noise
    else:
        raise ValueError(f"Unknown setting: {setting}")
    
    result[:, :200, :] = first_200_tokens
    result = result.to(device)
    return result


def get_all_sae_info(layer, sae_width, target_l0, model="gemma_2_9b", layer_type="res"):
    actual_l0 = get_l0_closest_to(get_gemma_sae_params_for_layer(layer, model=model, layer_type=layer_type)[sae_width], target_l0)
    sae_inf = get_sae_info_by_params(layer=str(layer), sae_width=sae_width, sae_l0=actual_l0, model=model, layer_type=layer_type)
    return sae_inf, actual_l0
# %%


if layer_type == "mlp":
    hook_names = [f"blocks.{layer}.hook_mlp_out" for layer in layers]
elif layer_type == "att":
    hook_names = [f"blocks.{layer}.attn.hook_z" for layer in layers]
elif layer_type == "res":
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
else:
    raise ValueError(f"Unknown layer type {layer_type}")

# %%

result_dict = get_gemma_sae_params_for_layer(layers[0], model="gemma_2_9b", layer_type="res")
all_widths = ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]
all_widths = all_widths[::-1]
all_l0s = result_dict["16k"]

target_layer = 20

all_settings = ["sae", "sae_and_linear", "sae_and_noise"]
all_baselines = [None] * len(all_settings)


if(loop_setting == "width"):
    loop_list = all_widths
    # Sort widths in ascending order
    
    target_l0 = 60
elif(loop_setting == "l0"):
    loop_list = all_l0s
    # Sort l0s in ascending order
    width = "16k"

# %%
import einops

all_outer_loop_losses = []

with torch.no_grad():
    for loop_ind, loop_parameter in enumerate(loop_list):
        print(f"Running for {loop_setting} {loop_parameter}")
        if(loop_setting == "width"):
            width = loop_parameter
        elif(loop_setting == "l0"):            
            target_l0 = loop_parameter

        resid_sae_info, resid_L0 = get_all_sae_info(layers[0], width, target_l0, model=gemma_model, layer_type="res")
        reshaped_reconstruction_vecs = einops.rearrange(resid_sae_info.reconstruction_vecs_flattened, "(n c) d -> n c d", c = 1023 - 200)
        # mlp_sae_info, mlp_L0 = get_all_sae_info(layers[0], width, target_l0, model=gemma_model, layer_type="mlp")
        # att_sae_info, att_L0 = get_all_sae_info(layers[0], width, target_l0, model=gemma_model, layer_type="att")

        sae_name = resid_sae_info.sae_name
        # print(sae_name)
        # sae, cfg_dict, sparsity = SAE.from_pretrained(
        #     release=f"gemma-scope-{size}-pt-{layer_type}",
        #     sae_id=sae_name,
        #     device = device
        # )
        # sae = sae.to(torch.float32).to(device)
            
        layer_specific_name = layer_type
        if layer_type=="res":
            layer_specific_name = "resid"


        linear_solution = torch.load(f"{BASE_DIR}/results/{layer_specific_name}_solution_{layers[0]}_{width}_{target_l0}.pt").to(torch.float32).to(device).to(torch.float16)
        mean = torch.load(f"{BASE_DIR}/results/{layer_specific_name}_mean_{layers[0]}_{width}_{target_l0}.pt").to(torch.float32).to(device).to(torch.float16)
        r_squared = torch.load(f"{BASE_DIR}/results/{layer_specific_name}_r_squared_{layers[0]}_{width}_{target_l0}.pt")
        r_squared_sae = torch.load(f"{BASE_DIR}/results/{layer_specific_name}_r_squared_sae_{layers[0]}_{width}_{target_l0}.pt")
        r_squared_linear = torch.load(f"{BASE_DIR}/results/{layer_specific_name}_r_squared_linear_{layers[0]}_{width}_{target_l0}.pt")
        r_squared_nonlinear = torch.load(f"{BASE_DIR}/results/{layer_specific_name}_r_squared_nonlinear_{layers[0]}_{width}_{target_l0}.pt")

        # Calculate percentage recovered for r_squared linear
        # Zero-point is sae, 100% is original
        ceiling_point = 1.0
        linear_r_squared_percent_recovered = ((r_squared_sae - r_squared_linear) / (r_squared_sae - ceiling_point)) * 100
        nonlinar_r_squared_percent_recovered = ((r_squared_sae - r_squared_nonlinear) / (r_squared_sae - ceiling_point)) * 100

        # Load Norm percentages
        percentage_of_norm_linear = torch.load(f"{BASE_DIR}/results/{layer_specific_name}_percentage_of_norm_linear_{layers[0]}_{width}_{target_l0}.pt")
        percentage_of_norm_nonlinear = torch.load(f"{BASE_DIR}/results/{layer_specific_name}_percentage_of_norm_nonlinear_{layers[0]}_{width}_{target_l0}.pt")


        # Main execution
        data = hf_dataset_to_generator(dataset_name)

        # all_settings = ["sae", "sae_and_linear", "sae_and_noise", "sae_and_error_hat_projection", "sae_and_noise_projection"]

        original_losses = []
        all_setting_losses = [[] for _ in all_settings]


        bar = tqdm(range(0, num_contexts, batch_size))
        # bar = range(0, num_contexts, batch_size)
        with torch.no_grad():
            for i in bar:
                # print("-------------------------------")
                input_tokens = tokenized_batch(data).to(device)
                
                # Run original model loss
                original_loss = model(input_tokens, return_type="loss", loss_per_token=True)
                original_loss = original_loss[:, 200:].mean()
                # print(original_loss)
                original_losses.append(original_loss.item())
                
                
                sae_vecs = reshaped_reconstruction_vecs[i].to(device)
                for setting_ind, setting in enumerate(all_settings):
                    hook_function = partial(inject_sae_baseline, sae_vecs=sae_vecs, linear_solution=linear_solution, setting=setting, baseline=all_baselines[setting_ind])
                    loss = model.run_with_hooks(input_tokens,
                                                fwd_hooks=[(hook_names[0], hook_function)],
                                                return_type="loss",
                                                loss_per_token=True
                    )
                    loss = loss[:, 200:].mean()
                    # print(loss)
                    all_setting_losses[setting_ind].append(loss.item())

        # all_settings = ["sae", "sae_and_linear", "sae_and_noise"]

        # Calculate percentage of loss explained: 
        # 0% = SAE
        # 100% = Original
        average_original_loss = sum(original_losses)/len(original_losses)
        average_sae_loss = sum(all_setting_losses[0])/len(all_setting_losses[0])
        average_linear_loss = sum(all_setting_losses[1])/len(all_setting_losses[1])
        average_noise_loss = sum(all_setting_losses[2])/len(all_setting_losses[2])
        
        linear_percent_recovered = ((average_sae_loss - average_linear_loss) / (average_sae_loss - average_original_loss)) * 100
        noise_percent_recovered = ((average_sae_loss - average_noise_loss) / (average_sae_loss - average_original_loss)) * 100
        print(f"Linear CE Percent Recovered: {linear_percent_recovered}")
        print(f"Linear R^2 Percent Recovered: {linear_r_squared_percent_recovered}")
        print(f"Linear Norm Percent Recovered: {percentage_of_norm_linear}")
        print(f"Nonlinear CE Percent Recovered: {noise_percent_recovered}")
        print(f"Nonlinear R^2 Percent Recovered: {nonlinar_r_squared_percent_recovered}")
        print(f"Nonlinear Norm Percent Recovered: {percentage_of_norm_nonlinear}")
        print(f"Original Loss: {average_original_loss}")
        print(f"SAE Loss: {average_sae_loss}")
        print(f"Linear Loss: {average_linear_loss}")
        print(f"Noise Loss: {average_noise_loss}")
        all_outer_loop_losses.append((width, resid_L0, linear_percent_recovered, noise_percent_recovered, linear_r_squared_percent_recovered, nonlinar_r_squared_percent_recovered, percentage_of_norm_linear, percentage_of_norm_nonlinear))

        # delete sae to save memory
        # del sae
        # clear cache w/ gc
        gc.collect()
        torch.cuda.empty_cache()

with open(f"{BASE_DIR}/results/ce_{layer_type}_{layers[0]}_{size}_{loop_setting}.pkl", "wb") as f:
    pickle.dump(all_outer_loop_losses, f)

    
# %%
