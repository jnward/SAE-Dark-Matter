# Read in pretrained.yaml
# %%

import yaml
import os
import pickle


with open("gemma_sae_pickles/pretrained.yaml", "r") as f:
    pretrained = yaml.load(f, Loader=yaml.FullLoader)



for size in ["2b", "9b"]:
    for extension in ["res", "mlp", "att"]:
        layer_to_width_l0_pairs = {}
        for sae in pretrained[f"gemma-scope-{size}-pt-{extension}"]["saes"]:
            # Parse id of form 'layer_12/width_262k/average_l0_67'
            layer, width, l0 = sae["id"].split("/")
            layer = int(layer.split("_")[1])
            width = width.split("_")[1]
            l0 = int(l0.split("_")[2])
            if layer not in layer_to_width_l0_pairs:
                layer_to_width_l0_pairs[layer] = []
            layer_to_width_l0_pairs[layer].append((width, l0))

        with open(f"gemma_sae_pickles/gemma_sae_dict_{size}_{extension}.pkl", "wb") as f:
            pickle.dump(layer_to_width_l0_pairs, f)