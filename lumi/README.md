# Running on Lumi

git clone https://github.com/luukkonenr/Megatron-DeepSpeed.git

## FlashAttention
At the moment our approach is a bit hacky. Easiest solution is to use one of our shared containers with FlashAttention already installed, otherwise you'll need to have an environment with a rocm-version more recent than LUMI currently offers and you'll need to play with containers. 

Installation approaches:
1) Use a container that has this ROCM, create a `singularity --sandbox <CONTAINER>`. You'll need to manually create empty folders to the locations you want to be visible from the container, minimally your home-directory or else these bindings will fail. e.g. `mkdir /path/to/my/container/$USER`

Enter the container with `singularity shell --writable <CONTAINER>`

2) Build a compatible container on a different system e.g. cPouta. I haven't been lucky with this one yet, so all contributions welcome!


## Tokenization
Grab the tokenizer (merges.txt and vocab.json) from a relevant path, in our case /project_id/tokenizers/<TOKENIZER_NAME>/. 
This repo differs from the upstream https://github.com/microsoft/Megatron-DeepSpeed.git only by changin `eod`-token from `<|endoftext|>` to `</s>`.
You can also use the upstream repo and apply the change manually.

## Running

Copy `launch.sh` and `pretrain_vikin_example.sh` to the project root and modify as needed.
