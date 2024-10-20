import pickle
import os

# Add parent dir to path
this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
os.sys.path.insert(0, parent_dir)

from utils import get_gemma_sae_params_for_layer, get_l0_closest_to


gemma_saes = get_gemma_sae_params_for_layer(layer=20, model="gemma_2_9b")

device = "cuda:0"

target_l0 = 60
with open("scripts/run_all.sh", "w") as f:
    for width in ["16k", "32k", "65k", "131k", "262k", "524k", "1m"]:
        l0s = gemma_saes[width]
        l0 = get_l0_closest_to(l0s, target_l0)
        f.write(
        f"python encoder_pursuit.py --layer 20 --width {width} --base_l0 {l0} --device {device}\n"
        )

width = "16k"

with open("scripts/run_all.sh", "a") as f:
    l0s = gemma_saes[width]
    sorted_l0s = sorted(l0s)
    for l0 in sorted_l0s:
        f.write(
            f"python encoder_pursuit.py --layer 20 --width {width} --base_l0 {l0} --device {device}\n"
        )
