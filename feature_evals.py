# %%
from utils import get_sae_info, run_lstsq
import os
import sys
from utils import BASE_DIR

from functools import partial
from nnsight import LanguageModel
import asyncio


import torch
import einops
import os

from functools import partial

import torch
from nnsight import LanguageModel

from sae_auto_interp.autoencoders.wrapper import AutoencoderLatents
from sae_auto_interp.autoencoders.OpenAI import Autoencoder


from sae_auto_interp.features import FeatureCache
from sae_auto_interp.features.features import FeatureRecord
from sae_auto_interp.utils import load_tokenized_data


from sae_auto_interp.features import FeatureDataset, pool_max_activation_windows, random_activation_windows, sample
from sae_auto_interp.config import FeatureConfig, ExperimentConfig


from sae_auto_interp.explainers import SimpleExplainer
from sae_auto_interp.scorers import RecallScorer
from tqdm import tqdm
import pickle


from sae_auto_interp.clients import OpenRouter, Local
from sae_auto_interp.utils import display
import argparse

from eleuther_sae_modified import sae

from sae_lens import SAE

os.makedirs(f"{BASE_DIR}/feature_evals", exist_ok=True)

# %%

layer =  12
sae_name = "layer_12/width_16k/average_l0_82"

info = get_sae_info(layer = layer, sae_name=sae_name, model="gemma_2_2b")
original_sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=f"gemma-scope-2b-pt-res",
    sae_id=sae_name,
    device = "cpu"
)

# Learn linear error probe
x = info.acts_flattened
y = info.sae_error_vecs_flattened
resids, r_squared, probe = run_lstsq(x, y, lstsq_token_threshold="all", device="cpu")



# %%

CTX_LEN = 128
BATCH_SIZE = 8
# N_TOKENS = 10_000_000
N_TOKENS  = 10_000_000
N_SPLITS = 2
NUM_FEATURES_TO_TEST = 1000


# Set torch seed
torch.manual_seed(0)

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

layer = 12

if not is_notebook:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--sae_type", type=str, required=True, choices=["linear", "nonlinear"])
    argparser.add_argument("--to_do", type=str, required=True, choices=["generate", "eval", "both"])
    argparser.add_argument("--device", type=str, default="cuda:1")

    args = argparser.parse_args()
    sae_type = args.sae_type
    to_do = args.to_do
    device = args.device
else:
    sae_type = "nonlinear"
    to_do = "generate"
    device = "cuda:1"

SAE_PATH = f"sae-ckpts/{sae_type}_error/model.layers.{layer}"

RAW_ACTIVATIONS_DIR = f"{BASE_DIR}/feature_evals/{SAE_PATH}"
SAVE_FILE = f"{BASE_DIR}/feature_evals/{SAE_PATH}/results.pkl"
FINAL_SAVE_FILE = f"{BASE_DIR}/feature_evals/{SAE_PATH}/final_results.pkl"

# %%

ae = sae.Sae.load_from_disk(SAE_PATH)
ae = ae.to(device)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

if to_do in ["generate", "both"]:
    model = LanguageModel("google/gemma-2-2b", device_map=device, dispatch=True)

tokens = load_tokenized_data(
    CTX_LEN,
    tokenizer,
    "kh4dien/fineweb-100m-sample",
    "train[:15%]",
)

# %%


WIDTH = len(ae.W_dec)
# Get NUM_FEATURES_TO_TEST random features to test without replacement
random_features = torch.randperm(WIDTH)[:NUM_FEATURES_TO_TEST]


# %%
generate = to_do in ["generate", "both"]
if generate:

    gpu_original_sae = original_sae.to(device)
    gpu_probe = probe.to(device)

    def _forward(ae, x):
        if sae_type == "nonlinear":
            x = x - gpu_original_sae(x) + x @ gpu_probe
        elif sae_type == "linear":
            x = x @ (-gpu_probe)
        else:
            raise ValueError("Invalid sae_type")
        batch_size = x.shape[0]
        flattened_x = einops.rearrange(x, "b c d -> (b c) d")
        forward_output = ae.forward(flattened_x)
        top_acts = forward_output.latent_acts
        top_indices = forward_output.latent_indices
        expanded = torch.zeros(top_acts.shape[0], WIDTH, device=device)
        expanded.scatter_(1, top_indices, top_acts)
        expanded = einops.rearrange(expanded, "(b c) d -> b c d", b=batch_size)
        return expanded

    # We can simply add the new module as an attribute to an existing
    # submodule on GPT-2's module tree.
    submodule = model.model.layers[layer]
    submodule.ae = AutoencoderLatents(
        ae, 
        partial(_forward, ae),
        width=WIDTH
    )

    with model.edit(" ", inplace=True):
        acts = submodule.output[0]
        submodule.ae(acts, hook=True)

    with model.trace("hello, my name is"):
        latents = submodule.ae.output.save()

    module_path = submodule.path

    submodule_dict = {module_path : submodule}
    module_filter = {module_path : random_features.to(device)}

    cache = FeatureCache(
        model, 
        submodule_dict, 
        batch_size=BATCH_SIZE, 
        filters=module_filter
    )

    cache.run(N_TOKENS, tokens)

    cache.save_splits(
        n_splits=N_SPLITS,
        save_dir=RAW_ACTIVATIONS_DIR,
    )

# %%


cfg = FeatureConfig(
    width = WIDTH,
    min_examples = 200,
    max_examples = 2_000,
    example_ctx_len = 128,
    n_splits = 2,
)

sample_cfg = ExperimentConfig(n_random=50)

# This is a hack because this isn't currently defined in the repo
sample_cfg.chosen_quantile = 0

dataset = FeatureDataset(
    raw_dir=RAW_ACTIVATIONS_DIR,
    cfg=cfg,
)
# %%

if to_do == "generate":
    exit()

# %%

# Define these functions here so we don't need to edit the functions in the git submodule
def default_constructor(
    record,
    tokens,
    buffer_output,
    n_random: int,
    cfg: FeatureConfig
):
    pool_max_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        cfg=cfg
    )

    random_activation_windows(
        record,
        tokens=tokens,
        buffer_output=buffer_output,
        n_random=n_random,
        ctx_len=cfg.example_ctx_len,
    )


constructor=partial(
    default_constructor,
    n_random=sample_cfg.n_random,
    tokens=tokens,
    cfg=cfg
)

sampler = partial(
    sample,
    cfg=sample_cfg
)


def load(
    dataset,
    constructor,
    sampler,
    transform = None
):
    def _process(buffer_output):
        record = FeatureRecord(buffer_output.feature)
        if constructor is not None:
            constructor(record=record, buffer_output=buffer_output)

        if sampler is not None:
            sampler(record)

        if transform is not None:
            transform(record)

        return record

    for buffer in dataset.buffers:
        for data in buffer:
            if data is not None:
                yield _process(data)
# %%

record_iterator = load(constructor=constructor, sampler=sampler, dataset=dataset, transform=None)

# next_record = next(record_iterator)

# display(next_record, tokenizer, n=5)
# %%

# Command to run: vllm serve hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --max_model_len 10000 --tensor-parallel-size 2
client = Local("hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4")
# %%

async def run_async():

    global record_iterator


    positive_scores = []
    negative_scores = []
    explanations = []
    feature_ids = []
    total_positive_score = 0
    total_negative_score = 0
    total_evaluated = 0

    bar = tqdm(record_iterator, total=NUM_FEATURES_TO_TEST)



    for record in bar:

        explainer = SimpleExplainer(
            client,
            tokenizer,
            # max_new_tokens=50,
            max_tokens=50,
            temperature=0.0
        )

        explainer_result = await explainer(record)
        # explainer_result = asyncio.run(explainer(record))

        # print(explainer_result.explanation)
        record.explanation = explainer_result.explanation


        scorer = RecallScorer(
            client,
            tokenizer,
            max_tokens=25,
            temperature=0.0,
            batch_size=4,
        )


        score = await scorer(record)

        quantile_positives = [0 for _ in range(11)]
        quantile_totals = [0 for _ in range(11)]
        negative_positives = 0
        negative_totals = 0
        for score_instance in score.score:
            quantile = score_instance.distance
            if quantile != -1 and score_instance.prediction != -1:
                quantile_totals[quantile] += 1
                if score_instance.prediction == 1:
                    quantile_positives[quantile] += 1
            if quantile == -1 and score_instance.prediction != -1:
                negative_totals += 1
                if score_instance.prediction == 1:
                    negative_positives += 1

        positive_scores.append((quantile_positives, quantile_totals))
        negative_scores.append((negative_positives, negative_totals))

        if (sum(quantile_totals) == 0) or (negative_totals == 0):
            continue

        total_positive_score += sum(quantile_positives) / sum(quantile_totals)
        total_negative_score += negative_positives / negative_totals
        total_evaluated += 1
        
        bar.set_description(f"Positive Recall: {total_positive_score / total_evaluated}, Negative Recall: {total_negative_score / total_evaluated}")

        print(quantile_positives, quantile_totals)

        explanations.append(record.explanation)

        feature_ids.append(record.feature.feature_index) 
        

        with open(SAVE_FILE, "wb") as f:
            pickle.dump((positive_scores, negative_scores, explanations, feature_ids), f)

    with open(FINAL_SAVE_FILE, "wb") as f:
        pickle.dump((positive_scores, negative_scores, explanations, feature_ids), f)


# Switch comment when running in notebook/command line
# await run_async()
asyncio.run(run_async())

# %%