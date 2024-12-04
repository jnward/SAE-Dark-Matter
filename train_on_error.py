# %%
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
import einops

from eleuther_sae_modified.sae import SaeConfig, SaeTrainer, TrainConfig
from eleuther_sae_modified.sae.data import chunk_and_tokenize
import argparse
from utils import get_sae_info, run_lstsq

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

# %%

if not is_notebook:

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_on", type=str, default="error", choices=["error", "linear_prediction_of_error", "nonlinear_error"])
    argparser.add_argument("--device", type=str, default="cuda:0")
    
    args = argparser.parse_args()
    train_on = args.train_on
    device = args.device
else:
    train_on = "error"
    # train_on = "linear_prediction_of_error"
    # train_on = "nonlinear_error"

    device = "cuda:0"

# %%

expansion_factor = 16
k = 32

layer =  12
sae_name = "layer_12/width_16k/average_l0_82"

torch.cuda.set_device(device)

# %%

info = get_sae_info(layer = layer, sae_name=sae_name, model="gemma_2_2b")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=f"gemma-scope-2b-pt-res",
    sae_id=sae_name,
    device = device
)


# %%

# Learn linear error probe
x = info.acts_flattened
y = info.sae_error_vecs_flattened
resids, r_squared, probe = run_lstsq(x, y, lstsq_token_threshold="all", device=device)
probe = probe.to(device).bfloat16()

# %%

MODEL = "google/gemma-2-2b"
dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)
dataset = dataset.shuffle(seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
max_seq_len = 1024
tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=max_seq_len)

# %%

total_train_tokens = 500_000_000
cutoff = (total_train_tokens + max_seq_len - 1) // max_seq_len
subset_train_data = tokenized.select(range(cutoff))

# %%

gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": device},
    torch_dtype=torch.bfloat16,
)

# %%

def remove_first_200_per_context(x):
    x = einops.rearrange(x, "(b c) d -> b c d", c=1024)
    x = x[:, 200:, :]
    x = einops.rearrange(x, "b c d -> (b c) d")
    return x


if train_on == "error":
    acts_lambda = lambda x: remove_first_200_per_context(x - sae(x))
elif train_on == "linear_prediction_of_error":
    acts_lambda = lambda x: remove_first_200_per_context(x @ (-probe))
elif train_on == "nonlinear_error":
    acts_lambda = lambda x: remove_first_200_per_context(x - sae(x) + x @ probe)



batch_size = 8
cfg = TrainConfig(
    SaeConfig(k=k, 
              expansion_factor=expansion_factor),
    batch_size=batch_size,
    layers=[layer],
    wandb_project="predicting_sae_error",
    run_name=train_on,
)
trainer = SaeTrainer(cfg, tokenized, gpt, hidden_modification=acts_lambda)
trainer.fit()

# %%
