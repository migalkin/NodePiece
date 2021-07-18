## NodePiece for OGB WikiKG 2

The code to run NodePiece on the OGB WikiKG 2 dataset.

* OGB repo: [github](https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/wikikg2)
* Decoder models taken from the AutoSF repo: [github](https://github.com/AutoML-4Paradigm/AutoSF/tree/AutoSF-OGB/wikikg2)
* Experiments executed on: Tesla V100 32GB, 64GB RAM 

NodePiece dramatically reduces the entity embeddings matrix and can be used with standard decoders from the OGB Leaderboard.


Paired with `AutoSF` scoring function, NodePiece yields the following performance being 70-180x smaller in #parameters (cf. [OGB WikiKG2 Leaderboard](https://ogb.stanford.edu/docs/leader_linkprop/)):

| Model | Vocabulary Size  |  Parameters  | Validation MRR | Test MRR |
| ----- | ------ | ----- | ----- | ----- |
| `NodePiece + AutoSF` | 20k  | 6,860,602 | 0.5806 &pm; 0.0047 | 0.5703 &pm; 0.0035 |
| `AutoSF` | 2.5M |  500,227,800 | 0.5510 &pm; 0.0063 | 0.5458 &pm; 0.0052 |
| `TransE (500d)` | 2.5M |  1,250,569,500 | 0.4272 &pm; 0.0030 | 0.4256 &pm; 0.0030 |

### Running the experiment
1. We have pre-computed a vocabulary of 20k anchor nodes (~910 MB). Download it using the `download.sh` script:
```bash
sh download.sh
```

2. Install the requirements from the `requirements.txt`

3. Run the code with the best hyperparameters using the main script
```bash
sh run_ogb.sh
```

### Vocabulary construction (Tokenization)


Alternatively, you can create a new vocab and run the tokenization with any number of anchors. 
For that, we improved the anchor mining procedure to be parallelizable and work in a batch fashion:
* We first run METIS partitioning available in [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric) so that anchor mining will be done independently for each partition. 
The number of partitions `--part` should ideally be equal or twice larger that the `--cpu_num` parameter.
* Then we tokenize `--tkn_batch` nodes in each iteration (eg, 500K nodes will create 5000 batches)  
* Note that this process might be RAM-hungry. Approximate requirements would be a server with 8 CPUs and 64 GB RAM.
* Tokenization time depends on several parameters (cf. the [main codebase](https://github.com/migalkin/NodePiece)). For the above server configuration, mining the vocab of 20k anchors might take 2-8 hours in the multiprocessing mode.