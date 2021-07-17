from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader

from ogb_wikikg2.model import KGEModel

from ogb_wikikg2.dataloader import TrainDataset
from ogb_wikikg2.dataloader import BidirectionalOneShotIterator

from ogb.linkproppred import LinkPropPredDataset, Evaluator
from collections import defaultdict
from tqdm import tqdm
import time
from tensorboardX import SummaryWriter

from ogb_tokenizer import NodePiece_OGB
from ogb_wikikg2.dummy_factory import DummyTripleFactory


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--dataset', type=str, default='ogbl-wikikg2', help='dataset name, default to wikikg2')
    parser.add_argument('--model', default='RotatE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=1, type=int)
    parser.add_argument('-d', '--hidden_dim', default=100, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=512, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0005, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=2, type=int)
    parser.add_argument('-randomSeed', default=0, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=10000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=1000, type=int)
    parser.add_argument('--valid_steps', default=1, type=int)
    parser.add_argument('--log_steps', default=1, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--print_on_screen', action='store_true', help='log on screen or not')
    parser.add_argument('--ntriples_eval_train', type=int, default=200000,
                        help='number of training triples to evaluate eventually')
    parser.add_argument('--neg_size_eval_train', type=int, default=500,
                        help='number of negative samples when evaluating training triples')

    parser.add_argument('--anchors', type=int, default=10000, help='Number of anchors to mine and use')
    parser.add_argument('--ancs_sp', type=int, default=50, help='Limit to topK of shortest paths per entity')

    parser.add_argument('--st_deg', type=float, default=0.4, help='Anchors: ratio of top degree nodes')
    parser.add_argument('--st_ppr', type=float, default=0.4, help='Anchors: ratio of top ppr nodes')
    parser.add_argument('--st_rand', type=float, default=0.2, help='Anchors: ratio of randomly selected nodes')
    parser.add_argument('--tkn_batch', type=int, default=100, help='Batch size for iGraph anchor mining')
    parser.add_argument('--inverses', action='store_true', help='whether to add inverse edges')
    parser.add_argument('--val_inverses', action='store_true', help='whether to add inverse edges to the validation set')
    parser.add_argument('--tkn_dir', type=str, default="all", help='neighbors direction for igraph')
    parser.add_argument('--part', type=int, default=1, help='tokenization on METIS graph partitions (for large graphs)')

    parser.add_argument('--pooler', type=str, default="cat", help='Set encoder')
    parser.add_argument('--rel_hash', type=str, default=None, help='Path encoder: avg or gru')
    parser.add_argument('--policy', type=str, default="sum", help='Sum or cat anchors and aggregated paths')
    parser.add_argument('--max_paths', type=int, default=10, help='How many paths per anchor to retain')
    parser.add_argument('--trf_layers', type=int, default=4, help='Num of transformer layers')
    parser.add_argument('--trf_heads', type=int, default=8, help='Num of transformer heads')
    parser.add_argument('--trf_hidden', type=int, default=512, help='Transformer FC size and REL encoder size')
    parser.add_argument('--drop', type=float, default=0.1, help='Dropout in layers')
    parser.add_argument('--use_dists', action='store_true', default=True, help='use path lengths as pos enc')
    parser.add_argument('--sample_rels', type=int, default=5, help='number of relations in the relational context')
    parser.add_argument('--noanc', action='store_true', help='ablation: no anchors')


    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    args.dataset = argparse_dict['dataset']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.anchor_embeddings.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'anchor_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics, writer):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        writer.add_scalar("_".join([mode, metric]), metrics[metric], step)


def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.evaluate_train):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)

    args.save_path = 'log/%s/%s/%s-%s/%s' % (
    args.dataset, args.model, args.hidden_dim, args.gamma, time.time()) if args.save_path == None else args.save_path
    writer = SummaryWriter(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    logging.info('Random seed: {}'.format(args.randomSeed))
    torch.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)

    dataset = LinkPropPredDataset(name=args.dataset)
    split_dict = dataset.get_edge_split()
    nentity = dataset.graph['num_nodes']
    nrelation = int(max(dataset.graph['edge_reltype'])[0]) + 1

    evaluator = Evaluator(name=args.dataset)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Dataset: %s' % args.dataset)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = split_dict['train']
    logging.info('#train: %d' % len(train_triples['head']))
    valid_triples = split_dict['valid']
    logging.info('#valid: %d' % len(valid_triples['head']))
    test_triples = split_dict['test']
    logging.info('#test: %d' % len(test_triples['head']))

    if args.inverses:
        # add inverse triples
        print("Adding inverse edges")
        orig_head, orig_tail = train_triples['head'], train_triples['tail']
        train_triples['head'] = np.concatenate([orig_head, orig_tail])
        train_triples['tail'] = np.concatenate([orig_tail, orig_head])
        train_triples['relation'] = np.concatenate([train_triples['relation'], train_triples['relation'] + nrelation])

        # let's add inverses to the validation
        if args.val_inverses:
            logging.info("Adding inverses to the validation set")
            orig_head, orig_tail = valid_triples['head'], valid_triples['tail']
            orig_head_negs, orig_tail_negs = valid_triples['head_neg'], valid_triples['tail_neg']
            valid_triples['head'] = np.concatenate([orig_head, orig_tail])
            valid_triples['tail'] = np.concatenate([orig_tail, orig_head])
            valid_triples['relation'] = np.concatenate([valid_triples['relation'], valid_triples['relation'] + nrelation])
            valid_triples['head_neg'] = np.concatenate([orig_head_negs, orig_tail_negs], axis=0)
            valid_triples['tail_neg'] = np.concatenate([orig_tail_negs, orig_head_negs], axis=0)

            logging.info("Adding inverses to the test set")
            orig_head, orig_tail = test_triples['head'], test_triples['tail']
            orig_head_negs, orig_tail_negs = test_triples['head_neg'], test_triples['tail_neg']
            test_triples['head'] = np.concatenate([orig_head, orig_tail])
            test_triples['tail'] = np.concatenate([orig_tail, orig_head])
            test_triples['relation'] = np.concatenate([test_triples['relation'], test_triples['relation'] + nrelation])
            test_triples['head_neg'] = np.concatenate([orig_head_negs, orig_tail_negs], axis=0)
            test_triples['tail_neg'] = np.concatenate([orig_tail_negs, orig_head_negs], axis=0)

            del orig_head, orig_tail, orig_head_negs, orig_tail_negs


        nrelation = nrelation * 2 + 1
    else:
        print("No inverse edges")
        nrelation += 1
    print(f"Total num relations: {nrelation}")
    # create a tokenizer based on train triples
    tokenizer = NodePiece_OGB(
        triples=DummyTripleFactory(train_triples, ne=nentity, nr=nrelation),
        anchor_strategy={
            "degree": args.st_deg,
            "betweenness": 0.0,
            "pagerank": args.st_ppr,
            "random": args.st_rand
        },
        num_anchors=args.anchors,
        num_paths=args.anchors,
        dataset_name=args.dataset,
        limit_shortest=args.ancs_sp,
        add_identity=False,
        mode="bfs",
        tkn_batch=args.tkn_batch,
        inv=args.inverses,
        dir=args.tkn_dir,
        partition=args.part,
        cpus=args.cpu_num
    )

    #if args.max_seq_len == 0 or args.max_seq_len != (tokenizer.max_seq_len + 3):
    max_seq_len = tokenizer.max_seq_len + 3  # as in the PathTrfEncoder, +1 CLS, +1 PAD, +1 LP tasks
    print(f"Set max_seq_len to {max_seq_len}")
    tokenizer.token2id[tokenizer.NOTHING_TOKEN] = len(tokenizer.token2id)


    train_count, train_true_head, train_true_tail = defaultdict(lambda: 4), defaultdict(list), defaultdict(list)
    for i in tqdm(range(len(train_triples['head']))):
        head, relation, tail = train_triples['head'][i], train_triples['relation'][i], train_triples['tail'][i]
        train_count[(head, relation)] += 1
        if not args.inverses:
            train_count[(tail, -relation - 1)] += 1
        train_true_head[(relation, tail)].append(head)
        train_true_tail[(head, relation)].append(tail)

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
        evaluator=evaluator,
        tokenizer=tokenizer,
        pooler=args.pooler,
        use_rels=args.rel_hash,
        rel_policy=args.policy,
        sample_paths=args.max_paths,
        trf_layers=args.trf_layers,
        trf_heads=args.trf_heads,
        trf_hidden=args.trf_hidden,
        drop=args.drop,
        use_distances=args.use_dists,
        max_seq_len=max_seq_len,
        sample_rels=args.sample_rels,
        triples=train_triples,
        ablate_anchors=args.noanc,
        device=torch.device('cuda') if args.cuda else torch.device('cpu'),
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    logging.info(f"Total number of params: {sum(p.numel() for p in kge_model.parameters())}")

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                         args.negative_sample_size, 'head-batch',
                         train_count, train_true_head, train_true_tail),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation,
                         args.negative_sample_size, 'tail-batch',
                         train_count, train_true_head, train_true_tail),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TrainDataset.collate_fn
        )

        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()),
            lr=current_learning_rate
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)
        training_logs = []
        max_val_mrr = 0
        best_val_metrics = None
        best_test_metrics = None
        best_metrics_step = 0

        # Training Loop
        for step in tqdm(range(init_step, args.max_steps)):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)
            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0 and step > 0:  # ~ 41 seconds/saving
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
                log_metrics('Train', step, metrics, writer)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0 and step > 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, args)
                log_metrics('Valid', step, metrics, writer)
                val_mrr = metrics['mrr_list']

                # evaluate on test set
                if val_mrr > max_val_mrr:
                    max_val_mrr = val_mrr
                    best_val_metrics = metrics
                    best_metrics_step = step

                    if args.do_test:
                        logging.info('Evaluating on Test Dataset...')
                        metrics = kge_model.test_step(kge_model, test_triples, args)
                        log_metrics('Test', step, metrics, writer)
                        best_test_metrics = metrics

        # record best metrics on validate and test set
        if args.do_valid and best_val_metrics != None:
            log_metrics('Best Val  Metrics', best_metrics_step, best_val_metrics, writer)
        if args.do_test and best_test_metrics != None:
            log_metrics('Best Test Metrics', best_metrics_step, best_test_metrics, writer)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, args)
        log_metrics('Valid', step, metrics, writer)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, args)
        log_metrics('Test', step, metrics, writer)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        small_train_triples = {}
        indices = np.random.choice(len(train_triples['head']), args.ntriples_eval_train, replace=False)
        for i in train_triples:
            small_train_triples[i] = train_triples[i][indices]
        metrics = kge_model.test_step(kge_model, small_train_triples, args, random_sampling=True)
        log_metrics('Train', step, metrics, writer)


if __name__ == '__main__':
    main(parse_args())