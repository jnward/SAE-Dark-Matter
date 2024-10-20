<img width="1853" alt="GPT-2 Auto-Discovered Multi-Dimensional Features" src="https://github.com/JoshEngels/MultiDimensionalFeatures/assets/15754392/cbe67ac3-feed-41a2-b31f-2a75406030da">

# Decomposing the Dark Matter of Sparse Autoencoders
This is the github repo for our paper "Decomposing the Dark Matter of Sparse Autoencoders".


## Reproducing each figure

Below are instructions to reproduce each figure (aspirationally). 

The required python packages to run this repo are
```
transformer_lens sae_lens transformers datasets torch
```
We recommend you create a new python venv named e.g. darkmatter and install these packages manually using pip:
```
python -m venv darkmatter
source darkmatter/bin/activate
pip install transformer_lens sae_lens transformers datasets torch
```
Let us know if anything does not work with this environment!


## Generating Activations

To run our experiments, you will need to generate and save model activations for all layers of interest on our token corpus, as well as SAE activations and metadata for all SAEs of interest.

### Generating Model Activations:



### Generating SAEInfoObjects:




## Contact

If you have any questions about the paper or reproducing results, feel free to email jengels@mit.edu.

## Note about SAE training implementation
For training topk SAEs we use a slightly modified version of https://github.com/EleutherAI/sae/tree/main stored in eleuther_sae_modified.