# %%
import pickle


layer = 20

layer_type = "att"

gemma_dict = pickle.load(open(f"gemma_sae_pickles/gemma_sae_dict_9b_{layer_type}.pkl", "rb"))

num_gpus = 1

gpu_offset = 1

hyperparams = gemma_dict[layer]

# Sort by width, which is of the form 16k or 1m or 256k etc.
values = []
for width, l0 in gemma_dict[layer]:
    if width[-1] == "k":
        width = int(width[:-1]) * 1000
    elif width[-1] == "m":
        width = int(width[:-1]) * 1000000
    values.append(width)
hyperparams = [x for _, x in sorted(zip(values, hyperparams))]

commands = []
for width, l0 in hyperparams:
    commands.append(
        f"python save_info_gemma.py --layer {layer} --sae_width {width} --sae_l0 {l0} --size 9b --layer_type {layer_type}"
    )

with open("scripts/run_all.sh", "w") as f:
    for i in range(0, len(commands), num_gpus):
        for device_id in range(num_gpus):
            if i + device_id < len(commands):
                f.write(f"{commands[i + device_id]} --device cuda:{device_id + gpu_offset} &\n")
        f.write("wait\n")