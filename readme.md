# NodePiece - Compositional and Parameter-Efficient Representations for Large Knowledge Graphs

<p align="center">
<img src="https://img.shields.io/badge/python-3.8-blue.svg">
<a href="https://github.com/migalkin/StarE/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
<a href="TODO"><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a>
<a href="TODO"><img src="https://img.shields.io/badge/Blog-Medium-03a87c"></a>
</p>

<img src="./NodePiece_fig.svg" width="100%">

NodePiece is a "tokenizer" for reducing entity vocabulary size in knowledge graphs. 
Instead of shallow embedding every node to a vector, we first "tokenize" each node by K anchor nodes and M relation types in its relational context.
Then, the resulting hash sequence is encoded through any injective function, e.g., MLP or Transformer.

Similar to Byte-Pair Encoding and WordPiece tokenizers commonly used in NLP, NodePiece can tokenize unseen nodes attached to the seen graph using the same anchor and relation vocabulary, which allows NodePiece to work out-of-the-box in the inductive settings using all the well-known scoring functions in the classical KG completion (like TransE or RotatE).
NodePiece also works with GNNs (we tested on node classification, but not limited to it, of course)

## NodePiece source code

The repo contains the code and experimental setups for reproducibility studies.

Each experiment resides in the respective folder:
* LP_RP - link prediction and relation prediction
* NC - node classification
* OOS_LP - out-of-sample link prediction

The repo is based on Python 3.8.
`wandb` is an optional requirement in case you have an existing account there and would like to track experimental results.
If you have a `wandb` account, the repo assumes you've performed
```
wandb login <your_api_key>
```
Using a GPU is recommended.

First, run a script which will download all the necessary pre-processed data and datasets. 
It takes approximately 1 GB.
```
sh download_data.sh
```

Install the dependencies in `requirements.txt`.
Note that when install Torch-Geometric you might want to use pre-compiled binaries for a certain version of python and torch. 
Check the manual [here](https://github.com/rusty1s/pytorch_geometric).

In the link prediction tasks, all the necessary datasets will be downloaded upon first script execution.

## Link Prediction 

The link prediction (LP) and relation prediction (RP) tasks use models, datasets, and evaluation protocols from [PyKEEN](https://github.com/pykeen/pykeen).

Navigate to the `lp_rp` folder: `cd lp_rp`

* Run the fb15k-237 experiment

```
python run_lp.py -loop lcwa -loss bce -b 512 -data fb15k237 -anchors 1000 -sp 100 -lr 0.0005 -ft_maxp 20 -pool cat -embedding 200 -sample_rels 15 -smoothing 0.4 -epochs 401
```

* Run the wn18rr experiment

```
python run_lp.py -loop slcwa -loss nssal -margin 15 -b 512 -data wn18rr -anchors 500 -sp 100 -lr 0.0005 -ft_maxp 50 -pool cat -embedding 200 -negs 20 -subbatch 2000 -sample_rels 4 -epochs 601
```

* Run the codex-l experiment

```
python run_lp.py -loop lcwa -loss bce -b 256 -data codex_l -anchors 7000 -sp 100 -lr 0.0005 -ft_maxp 20 -pool cat -embedding 200 -subbatch 10000 -sample_rels 6 -smoothing 0.3 -epochs 120
```

* Run the yago 3-10 experiment

```
python run_lp.py -loop slcwa -loss nssal -margin 50 -b 512 -data yago -anchors 10000 -sp 100 -lr 0.00025 -ft_maxp 20 -pool cat -embedding 200 -subbatch 2000 -sample_rels 5 -negs 10 -epochs 601
```

### Test evaluation reproducibility patch

PyKEEN 1.0.5 used in this repo has been identified to have issues at the filtering stage when evaluating on the test set.
In order to fully reproduce the reported test set numbers for transductive LP/RP experiments from the paper and resolve this issue, please apply the patch from the `lp_rp/patch` folder:
1. Locate pykeen in your environment installation:
```
<path_to_env>/lib/python3.<NUMBER>/site-packages/pykeen
```
2. Replace the `evaluation/evaluator.py` with the one from the `patch` folder
```bash
cp ./lp_rp/patch/evaluator.py <path_to_env>/lib/python3.<NUMBER>/site-packages/pykeen/evaluation/
```
3. Replace the `stoppers/early_stopping.py` with the one from the `patch` folder
```bash
cp ./lp_rp/patch/early_stopping.py <path_to_env>/lib/python3.<NUMBER>/site-packages/pykeen/stoppers/
```

This won't be needed once we port the codebase to newest versions of PyKEEN (1.4.0+) where this was fixed

## Relation Prediction

The setup is very similar to that of link prediction (LP) but we predict relations `(h,?,t)` now.

Navigate to the `lp_rp` folder: `cd lp_rp`

* Run the fb15k-237 experiment

```
python run_lp.py -loop slcwa -loss nssal -b 512 -data fb15k237 -anchors 1000 -sp 100 -lr 0.0005 -ft_maxp 20 -margin 15 -subbatch 2000 -pool cat -embedding 200 -negs 20 -sample_rels 15 -epochs 21 --rel-prediction True
```

* Run the wn18rr experiment

```
python run_lp.py -loop slcwa -loss nssal -b 512 -data wn18rr -anchors 500 -sp 100 -lr 0.0005 -ft_maxp 50 -margin 12 -subbatch 2000 -pool cat -embedding 200 -negs 20 -sample_rels 4 -epochs 151 --rel-prediction True
```

* Run the yago 3-10 experiment

```
python run_lp.py -loop slcwa -loss nssal -b 512 -data yago -anchors 10000 -sp 100 -lr 0.0005 -ft_maxp 20 -margin 25 -subbatch 2000 -pool cat -embedding 200 -negs 20 -sample_rels 5 -epochs 7 --rel-prediction True
```

## Node Classification

Navigate to the `nc` folder: `cd nc`

If you have a GPU, use `DEVICE cuda` otherwise `DEVICE cpu`.

The run on 5% of labeled data:

```
python run_nc.py DATASET wd50k MAX_QPAIRS 3 STATEMENT_LEN 3 LABEL_SMOOTHING 0.1 EVAL_EVERY 5 DEVICE cpu WANDB False EPOCHS 4001 GCN_HID_DROP2 0.5 GCN_HID_DROP 0.5 GCN_FEAT_DROP 0.5 EMBEDDING_DIM 100 GCN_GCN_DIM 100 LEARNING_RATE 0.001 GCN_ATTENTION True GCN_GCN_DROP 0.3 GCN_ATTENTION_DROP 0.3 GCN_LAYERS 3 DS_TYPE transductive MODEL_NAME stare TR_RATIO 0.05 USE_FEATURES False TOKENIZE True NUM_ANCHORS 50 MAX_PATHS 10 USE_TEST True
```

The run on 10% of labeled data:
```
python run_nc.py DATASET wd50k MAX_QPAIRS 3 STATEMENT_LEN 3 LABEL_SMOOTHING 0.1 EVAL_EVERY 5 DEVICE cpu WANDB False EPOCHS 4001 GCN_HID_DROP2 0.5 GCN_HID_DROP 0.5 GCN_FEAT_DROP 0.5 EMBEDDING_DIM 100 GCN_GCN_DIM 100 LEARNING_RATE 0.001 GCN_ATTENTION True GCN_GCN_DROP 0.3 GCN_ATTENTION_DROP 0.3 GCN_LAYERS 3 DS_TYPE transductive MODEL_NAME stare TR_RATIO 0.1 USE_FEATURES False TOKENIZE True NUM_ANCHORS 50 MAX_PATHS 10 USE_TEST True
```


## Out-of-sample Link Prediction

Navigate to the `oos_lp` folder: `cd oos_lp/src`

* Run the oos fb15k-237 experiment

```
python main.py -dataset FB15k-237 -model_name DM_NP_fb -ne 41 -lr 0.0005 -emb_dim 200 -batch_size 256 -simulated_batch_size 256 -save_each 100 -tokenize True -opt adam -pool trf -use_custom_reg False -reg_lambda 0.0 -loss_fc spl -margin 15 -neg_ratio 5 -wandb False -eval_every 20 -anchors 1000 -sample_rels 15
```

* Run the oos yago3-10 experiment

```
python main.py -dataset YAGO3-10 -model_name DM_NP_yago -ne 41 -lr 0.0005 -emb_dim 200 -batch_size 256 -simulated_batch_size 256 -save_each 100 -tokenize True -opt adam -pool trf -use_custom_reg False -reg_lambda 0.0 -loss_fc spl -margin 15 -neg_ratio 5 -wandb False -eval_every 20 -anchors 10000 -sample_rels 5
```