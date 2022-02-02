# Inductive Link Prediction with NodePiece

The task is coined in the paper by Teru et al [Inductive relation prediction by subgraph reasoning](https://arxiv.org/abs/1911.06962).
Training is executed on one graph, but inference link prediction is executed on a completely new graph
with unseen entities (but known relations).

For each KG dump (FB15k-237, WN18RR, NELL995) there exist 4 versions of the datasets 
varying in the size of transductive training and inductive inference graphs.

The NodePiece setup here does not use any anchors as inference graphs are disconnected from training graphs.
That is, we only use unique relations for node tokenization.

### Results

<table>
    <tr>
        <th rowspan="2">Dataset</th>
        <th colspan="4">HITS@10 (50 samples)</th>
    </tr>
    <tr>
        <th>v1</th>
        <th>v2</th>
        <th>v3</th>
        <th>v4</th>
    </tr>
    <tr>
        <th>FB15k-237</th>
        <td>0.873</td>
        <td>0.939</td>
        <td>0.944</td>
        <td>0.949</td>
    </tr>
    <tr>
        <th>WN18RR</th>
        <td>0.830</td>
        <td>0.886</td>
        <td>0.785</td>
        <td>0.807</td>
    </tr>
    <tr>
        <th>NELL</th>
        <td>0.890</td>
        <td>0.901</td>
        <td>0.936</td>
        <td>0.893</td>
    </tr>
</table>

Reproduce the reported experiments with the following commands. Add `-wandb True` if you want to log the results to WANDB.

### FB15k-237
* Version 1
```
python run_ilp.py -loss nssal -margin 25 -epochs 2503 -lr 0.0001 -data fb15k237 -sample_rels 12 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3
```
* Version 2
```
python run_ilp.py -loss nssal -margin 15 -epochs 2000 -lr 0.0001 -data fb15k237 -sample_rels 12 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3 -pna False -residual False -jk False -ind_v 2 -eval_bs 2048
```
* Version 3
```
python run_ilp.py -loss nssal -margin 15 -epochs 2000 -lr 0.0001 -data fb15k237 -sample_rels 12 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3 -pna False -residual False -jk False -ind_v 3 -eval_bs 2048
```
* Version 4
```
python run_ilp.py -loss nssal -margin 25 -epochs 2000 -lr 0.0001 -data fb15k237 -sample_rels 12 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3 -pna False -residual False -jk False -ind_v 4 -eval_bs 2048
```

### WN18RR
* Version 1
```
python run_ilp.py -loss nssal -margin 15 -epochs 589 -lr 0.0001 -data wn18rr -sample_rels 4 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3 -pna True -residual True
```
* Version 2
```
python run_ilp.py -loss nssal -margin 15 -epochs 2000 -lr 0.0001 -data wn18rr -sample_rels 4 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 6 -pna False -residual True -jk True -ind_v 2
```
* Version 3
```
python run_ilp.py -loss nssal -margin 5 -epochs 211 -lr 0.0001 -data wn18rr -sample_rels 4 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 6 -pna False -residual True -jk True -ind_v 3 -eval_bs 2048
```
* Version 4
```
python run_ilp.py -loss nssal -margin 20 -epochs 2000 -lr 0.0001 -data wn18rr -sample_rels 3 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 10 -pna False -residual True -jk True -ind_v 4 -eval_bs 2048
```

### NELL 995
* Version 1
```
python run_ilp.py -loss nssal -margin 15 -epochs 451 -lr 0.0001 -data nell -sample_rels 4 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3 -ind_v 1 -eval_bs 2048
```
* Version 2
```
python run_ilp.py -loss nssal -margin 20 -epochs 2000 -lr 0.0001 -data nell -sample_rels 6 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 4 -ind_v 2 -eval_bs 2048
```
* Version 3
```
python run_ilp.py -loss nssal -margin 30 -epochs 2000 -lr 0.0001 -data nell -sample_rels 4 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3 -ind_v 3 -eval_bs 2048
```
* Version 4
```
python run_ilp.py -loss nssal -margin 20 -epochs 2000 -lr 0.0001 -data nell -sample_rels 30 -negs 32 -gnn True -rp False -gnn_att True -gnn_layers 3 -ind_v 4 -eval_bs 2048 -pool trf
```

### More
More CLI arguments can be found in `run_ilp.py`, eg, you can start the relation prediction task with the `-rp True` flag.
