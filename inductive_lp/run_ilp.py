from pykeen.datasets import FB15k237, WN18RR, YAGO310
from pykeen.utils import resolve_device
from pykeen.losses import BCEWithLogitsLoss, SoftplusLoss, MarginRankingLoss, NSSALoss
from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop
from pykeen.trackers import WANDBResultTracker
from pykeen.evaluation import RankBasedEvaluator
from pykeen.sampling import BasicNegativeSampler
from pykeen.stoppers import EarlyStopper
from pykeen.models import RotatE
from torch.optim import Adam

from model.nodepiece_rotate import NodePieceRotate
from nodepiece_tokenizer import NodePiece_Tokenizer
from loops.inductive_slcwa import InductiveSLCWATrainingLoop
from loops.relation_rank_evaluator import RelationPredictionRankBasedEvaluator
from loops.ilp_evaluator import ILPRankBasedEvaluator
from utils.sample_negatives import sample_negatives

from datasets.ind_dataset import Ind_FB15k237, Ind_WN18RR, Ind_NELL

import torch
import click
import numpy as np
import wandb
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@click.command()
@click.option('-embedding', '--embedding-dimension', type=int, default=100)  # embedding dim for anchors and relations
@click.option('-loss', '--loss_fc', type=str, default='nssal')
@click.option('-loop', '--loop', type=str, default='slcwa')  # slcwa - negative sampling, lcwa - 1-N scoring
@click.option('-trf_hidden', '--transformer-hidden-dim', type=int, default=200)
@click.option('-trf_heads', '--transformer-num-heads', type=int, default=4)
@click.option('-trf_layers', '--transformer-layers', type=int, default=2)
@click.option('-trf_drop', '--transformer-dropout', type=float, default=0.1)
@click.option('-b', '--batch-size', type=int, default=512)
@click.option('-eval_bs', '--eval_bs', type=int, default=512)
@click.option('-epochs', '--num-epochs', type=int, default=1000)
@click.option('-lr', '--learning-rate', type=float, default=0.0005)
@click.option('-wandb', '--enable_wandb', type=bool, default=False)
@click.option('-data', '--dataset_name', type=str, default='wn18rr')
@click.option('-eval_every', '--eval_every', type=int, default=1)
@click.option('-ft_maxp', '--ft_max_paths', type=int, default=100)  # max anchor per node, should be <= total N of anchors
@click.option('-anc_dist', '--use_anchor_distances', type=bool, default=False)  # whether to add anchor distances
@click.option('-margin', '--margin', type=float, default=15)
@click.option('-max_seq_len', '--max_seq_len', type=int, default=0)
@click.option('-pool', '--pooling', type=str, default="cat")  # available encoders: "cat" or "trf"
@click.option('-subbatch', '--trf_subbatch', type=int, default=3000)
@click.option('-negs', '--num_negatives_ent', type=int, default=1)  # number of negative samples when training LP in sLCWA
@click.option('-smoothing', '--lbl_smoothing', type=float, default=0.0)  # label smoothing in the 1-N setup
@click.option('-relpol', '--rel_policy', type=str, default="sum")
@click.option('-rand_hashes', '--random_hashing', type=int, default=0)  # for ablations: use only random numbers as hashes
@click.option('-nn', '--nearest_neighbors', type=bool, default=True)  # use only nearest anchors per node
@click.option('-sample_rels', '--sample_rels', type=int, default=4)  # size of the relational context M
@click.option('-tkn_mode', '--tkn_mode', type=str, default="bfs")  # mining paths in iGRAPH,
@click.option('-no_anc', '--ablate_anchors', type=bool, default=True)  # don't use any anchors in hashes, keep only the relational context
@click.option('-ind_v', '--ind_v', type=int, default=1)  # 1 / 2 / 3 / 4 - which version of GraIL splits to use
@click.option('-rp', '--rp', type=bool, default=False)  # turn on for the relation prediction task
@click.option('-gnn', '--gnn', type=bool, default=False)  # whether to use a GNN encoder on top of NodePiece features
@click.option('-gnn_att', '--gnn_att', type=bool, default=False)  # GNN with attentional aggregation
@click.option('-gnn_lrga', '--gnn_lrga', type=bool, default=False)  # Low-Rank Global Attention
@click.option('-gnn_layers', '--gnn_layers', type=int, default=2)
@click.option('-gnn_att_drop', '--gnn_att_drop', type=float, default=0.1)
@click.option('-ilp_eval', '--ilp_eval', type=bool, default=True)  # True stands for evaluation of 50 random samples to replicate the Teru et al setup, otherwise eval against all entities
@click.option('-pna', '--pna', type=bool, default=False)  # whether to use Principal Neighborhood Aggregation
@click.option('-residual', '--residual', type=bool, default=False)  # whether to use residual connections for deep GNNs
@click.option('-jk', '--jk', type=bool, default=False)  # JK connections for GNN
def main(
        embedding_dimension,
        loss_fc,
        loop,
        transformer_hidden_dim: int,
        transformer_num_heads: int,
        transformer_layers: int,
        transformer_dropout: float,
        batch_size: int,
        eval_bs: int,
        num_epochs: int,
        learning_rate: float,
        enable_wandb: bool,
        dataset_name: str,
        eval_every: int,
        ft_max_paths: int,
        use_anchor_distances: bool,
        margin: float,
        max_seq_len: int,
        pooling: str,
        trf_subbatch: int,
        num_negatives_ent: int,
        lbl_smoothing: float,
        rel_policy: str,
        random_hashing: int,
        nearest_neighbors: bool,
        sample_rels: int,
        tkn_mode: str,
        ablate_anchors: bool,
        ind_v: int,
        rp: bool,
        gnn: bool,
        gnn_att: bool,
        gnn_lrga: bool,
        gnn_layers: int,
        gnn_att_drop: float,
        ilp_eval: bool,
        pna: bool,
        residual: bool,
        jk: bool,
):
    # Standard dataset loading procedures, inverses are necessary for reachability of nodes
    if dataset_name == 'fb15k237':
        dataset = Ind_FB15k237(create_inverse_triples=True, version=ind_v)
    elif dataset_name == 'wn18rr':
        dataset = Ind_WN18RR(create_inverse_triples=True, version=ind_v)
    elif dataset_name == "nell":
        dataset = Ind_NELL(create_inverse_triples=True, version=ind_v)

    negative_sampler_cls = BasicNegativeSampler
    negative_sampler_kwargs = dict(num_negs_per_pos=num_negatives_ent)
    loop_type = InductiveSLCWATrainingLoop

    training_triples_factory = dataset.transductive_part
    inference_triples_factory = dataset.inductive_inference

    # No need for anchor tokenization in inductive datasets
    # We will do relation-only tokenization inside the model
    kg_tokenizer = None
    device = resolve_device()

    # selecting the loss function
    if loss_fc == "softplus":
        ft_loss = SoftplusLoss()
    elif loss_fc == "bce":
        ft_loss = BCEWithLogitsLoss()
    elif loss_fc == "mrl":
        ft_loss = MarginRankingLoss(margin=margin)
    elif loss_fc == "nssal":
        ft_loss = NSSALoss(margin=margin)

    # Create the model
    finetuning_model = NodePieceRotate(embedding_dim=embedding_dimension, device=device, loss=ft_loss,
                                     triples=training_triples_factory, inference_triples=inference_triples_factory,max_paths=ft_max_paths, subbatch=trf_subbatch,
                                     max_seq_len=max_seq_len, tokenizer=kg_tokenizer, pooler=pooling,
                                     hid_dim=transformer_hidden_dim, num_heads=transformer_num_heads,
                                     use_distances=use_anchor_distances, num_layers=transformer_layers, drop_prob=transformer_dropout,
                                     rel_policy=rel_policy, random_hashes=random_hashing, nearest=nearest_neighbors,
                                     sample_rels=sample_rels, ablate_anchors=ablate_anchors, tkn_mode=tkn_mode, gnn=gnn, gnn_att=gnn_att,
                                     use_lrga=gnn_lrga, gnn_layers=gnn_layers, gnn_att_drop=gnn_att_drop, pna=pna, residual=residual, jk=jk)

    optimizer = Adam(params=finetuning_model.parameters(), lr=learning_rate)
    print(f"Number of params: {sum(p.numel() for p in finetuning_model.parameters())}")

    if loop == "slcwa":
        ft_loop = loop_type(model=finetuning_model, optimizer=optimizer, negative_sampler_cls=negative_sampler_cls,
                                    negative_sampler_kwargs=negative_sampler_kwargs)
    else:
        ft_loop = LCWATrainingLoop(model=finetuning_model, optimizer=optimizer)

    # add the results tracker if requested
    if enable_wandb:
        project_name = "NodePiece_ILP"

        tracker = WANDBResultTracker(project=project_name, group=None, settings=wandb.Settings(start_method='fork'))
        tracker.wandb.config.update(click.get_current_context().params)
    else:
        tracker = None

    if rp:
        valid_evaluator = RelationPredictionRankBasedEvaluator()
    else:
        if not ilp_eval:
            valid_evaluator = RankBasedEvaluator
        else:
            # GraIL-style evaluation against 50 random negatives
            head_samples, tail_samples = sample_negatives(valid_triples=dataset.inductive_val, all_pos=dataset.inductive_inference)
            valid_evaluator = ILPRankBasedEvaluator(head_samples=head_samples, tail_samples=tail_samples)
    valid_evaluator.batch_size = eval_bs

    # we don't actually use the early stopper here by setting the patience to 100000
    early_stopper = EarlyStopper(
        model=finetuning_model,
        relative_delta=0.0005,
        evaluation_triples_factory=dataset.inductive_val,
        frequency=eval_every,
        patience=100000,
        result_tracker=tracker,
        evaluation_batch_size=eval_bs,
        evaluator=valid_evaluator,
    )

    # Train LP / RP
    if loop == "lcwa":
        ft_loop.train(num_epochs=num_epochs, batch_size=batch_size, result_tracker=tracker,
                      stopper=early_stopper, label_smoothing=lbl_smoothing)
    else:
        ft_loop.train(num_epochs=num_epochs, batch_size=batch_size, result_tracker=tracker,
                      stopper=early_stopper)

    # run the final test eval
    if rp:
        test_evaluator = RelationPredictionRankBasedEvaluator()
    else:
        if not ilp_eval:
            test_evaluator = RankBasedEvaluator
        else:
            # test evaluator
            head_samples, tail_samples = sample_negatives(valid_triples=dataset.inductive_test,
                                                          all_pos=dataset.inductive_inference)
            test_evaluator = ILPRankBasedEvaluator(head_samples=head_samples, tail_samples=tail_samples)
    test_evaluator.batch_size = eval_bs

    # test_evaluator
    metric_results = test_evaluator.evaluate(
        model=finetuning_model,
        mapped_triples=dataset.inductive_test.mapped_triples,
        additional_filter_triples=[dataset.inductive_inference.mapped_triples, dataset.inductive_val.mapped_triples],
        use_tqdm=True,
        batch_size=eval_bs,
    )

    # log final results
    if enable_wandb:
        tracker.log_metrics(
            metrics=metric_results.to_flat_dict(),
            step=num_epochs + 1,
            prefix='test',
        )

    print("Test results")
    print(metric_results)


if __name__ == '__main__':
    main()
