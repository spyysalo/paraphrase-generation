# Paraphrase generation

Tools and resources for training causal language model for paraphrase
generation

## Data preparation

Download Turku paraphrase corpus data

```
mkdir source-data
wget -P source-data https://github.com/TurkuNLP/Turku-paraphrase-corpus/raw/ver-1.1.0/data-fi/train.json
wget -P source-data https://github.com/TurkuNLP/Turku-paraphrase-corpus/raw/ver-1.1.0/data-fi/dev.json
```

Convert to simple JSONL format

```
python3 scripts/get_turku_paraphrases.py source-data/train.json > train.jsonl
python3 scripts/get_turku_paraphrases.py source-data/dev.json > valid.jsonl
```

## Running on LUMI

First, load modules

```
module load cray-python
module load LUMI/22.08 partition/G rocm/5.1.4
```

Create virtual environment

```
python -m venv venv
source venv/bin/activate
```

Install python modules

```
python -m pip install --upgrade pip setuptools wheel
python3 -m pip install torch==1.12.1+rocm5.1.1 --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
python -m pip install --upgrade transformers datasets evaluate
```

Edit `scripts/gpu-sinteractive.sh` to use the right `--account`

Start interactive session on GPU node

```
./scripts/gpu-sinteractive.sh
source scripts/node-setup.sh
```

```
python3 train.py TurkuNLP/gpt3-finnish-3B --batch-size 4 --gradient-accumulation-steps 2 train.jsonl valid.jsonl
```
