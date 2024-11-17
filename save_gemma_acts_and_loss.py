# %%

import transformer_lens
from datasets import load_dataset
import torch
from tqdm import tqdm
import os
import argparse
from utils import BASE_DIR

torch.set_grad_enabled(False)


# %%


try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False

if not is_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--device", type=str, default="cuda:0")
    argparser.add_argument("--layers", type=int, nargs="+", default=list(range(41)))
    argparser.add_argument("--layer_type", choices=["res", "mlp", "att"], default="res")

    args = argparser.parse_args()

    device = args.device
    layers = args.layers
    layer_type = args.layer_type

else:

    device = "cuda:0"
    layers = [20]
    layer_type = "res"

# %%

size = "2b"

model = transformer_lens.HookedTransformer.from_pretrained(
    f"google/gemma-2-{size}",
    center_writing_weights=False,
    center_unembed=False,
    device=device,
)


# %%

if layer_type == "mlp":
    hook_names = [f"blocks.{layer}.hook_mlp_out" for layer in layers]
elif layer_type == "att":
    hook_names = [f"blocks.{layer}.attn.hook_z" for layer in layers]
elif layer_type == "res":
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
else:
    raise ValueError(f"Unknown layer type {layer_type}")

ctx_len = 1024
batch_size = 1
num_contexts = 300

# %%


dataset_name = "monology/pile-uncopyrighted"
save_dir_base = f"{BASE_DIR}/gemma_{size}_sae_scaling"
if layer_type == "mlp":
    save_dir_base = os.path.join(save_dir_base + "_mlp")
elif layer_type == "att":
    save_dir_base = os.path.join(save_dir_base + "_att")
os.makedirs(save_dir_base, exist_ok=True)

# %%


def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    dataset = dataset.shuffle(seed=42, buffer_size=10000)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()


data = hf_dataset_to_generator(dataset_name)


def tokenized_batch():
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


def get_activations_and_loss(input_tokens):
    losses, cache = model.run_with_cache(
        input_tokens, names_filter=lambda name: name in hook_names, return_type="loss", loss_per_token=True
    )
    acts = [cache[hook_name] for hook_name in hook_names]
    return losses, acts


# %%

all_model_losses = []
all_acts = [[] for _ in layers]
all_act_norms = [[] for _ in layers]
all_tokens = []

# %%


def save_so_far():
    all_model_losses_cat = torch.cat(all_model_losses, dim=0)
    all_tokens_cat = torch.cat(all_tokens, dim=0)
    torch.save(all_model_losses_cat, os.path.join(save_dir_base, "model_losses.pt"))
    torch.save(all_tokens_cat, os.path.join(save_dir_base, "tokens.pt"))

    for layer, layer_acts in zip(layers, all_acts):
        layer_acts_cat = torch.cat(layer_acts, dim=0)
        # torch.save(layer_acts_cat, os.path.join(save_dir_base, f"acts_layer_{layer}.pt"))
        layer_acts_cat.numpy().tofile(os.path.join(save_dir_base, f"acts_layer_{layer}.npy"))

    for layer, layer_act_norms in zip(layers, all_act_norms):
        layer_act_norms_cat = torch.cat(layer_act_norms, dim=0)
        torch.save(layer_act_norms_cat, os.path.join(save_dir_base, f"act_norms_layer_{layer}.pt"))


# %%


bar = tqdm(range(0, num_contexts, batch_size))
for i in bar:

    input_tokens = tokenized_batch().to(device)
    losses, acts = get_activations_and_loss(input_tokens)
    
    for index, (layer, layer_acts) in enumerate(zip(layers, acts)):
        layer_act_norms = layer_acts.norm(dim=-1)
        all_acts[index].append(layer_acts.to("cpu"))
        all_act_norms[index].append(layer_act_norms.to("cpu"))

    all_model_losses.append(losses.to("cpu"))
    all_tokens.append(input_tokens.to("cpu"))

save_so_far()

# %%
