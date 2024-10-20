# %%
import pickle
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--num_gpus", type=int, default=1)
argparser.add_argument("--gpu_offset", type=int, default=0)
args = argparser.parse_args()

num_gpus = args.num_gpus
gpu_offset = args.gpu_offset

# %%

layer_type = "res"

gemma_dict = pickle.load(open(f"gemma_sae_pickles/gemma_sae_dict_9b_{layer_type}.pkl", "rb"))

width = "131k"

target_l0 = 60

actual_l0s = []

for layer in range(40):
    hyperparams = gemma_dict[layer]
    l0s = [l0 for w, l0 in hyperparams if w == width]
    closest_l0_to_target = min(l0s, key=lambda x: abs(x - target_l0))
    actual_l0s.append(closest_l0_to_target)

# %%

commands = []
for layer, l0 in zip(range(40), actual_l0s):
    commands.append(
        f"python save_info_gemma.py --layer {layer} --sae_width {width} --sae_l0 {l0} --size 9b --layer_type {layer_type}"
    )

with open("scripts/run_all.sh", "w") as f:
    for i in range(0, len(commands), num_gpus):
        for device_id in range(num_gpus):
            if i + device_id < len(commands):
                f.write(f"{commands[i + device_id]} --device cuda:{device_id + gpu_offset} &\n")
        f.write("wait\n")