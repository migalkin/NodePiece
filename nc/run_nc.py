"""
The code is largely based on the original StarE codebase https://github.com/migalkin/StarE, but adapted for node classification
"""

import os

os.environ['MKL_NUM_THREADS'] = '1'

from functools import partial
import random
import wandb
import sys
import collections

from pathlib import Path
from utils.utils import *
from utils.utils_mytorch import parse_args, BadParameters, mt_save_dir
from loops.evaluation import eval_classification
from models_nc import StarE_PyG_NC
from models.nc_baselines import MLP, MLP_PyG
from loops.sampler import NodeClSampler
from loops.loops import training_loop_pyg_nc
from data_loaders.clean_datasets import load_clean_pyg
from utils.nodepiece_tokenizer import NodePiece_Tokenizer

"""
    CONFIG Things
"""

# Clamp the randomness
np.random.seed(42)
random.seed(42)
torch.manual_seed(132)


DEFAULT_CONFIG = {
    'BATCH_SIZE': 512,
    'CORRUPTION_POSITIONS': [0, 2],
    'DATASET': 'wd50k',
    'DEVICE': 'cpu',
    'EMBEDDING_DIM': 50,
    'FEATURE_DIM': 1024,
    'ENT_POS_FILTERED': True,
    'EPOCHS': 1000,
    'EVAL_EVERY': 1,
    'LEARNING_RATE': 0.0002,
    'MARGIN_LOSS': 5,
    'MAX_QPAIRS': 3,
    'MODEL_NAME': 'stare',
    'NUM_FILTER': 5,
    'RUN_TESTBENCH_ON_TRAIN': True,
    'SAVE': False,
    'SELF_ATTENTION': 0,
    'STATEMENT_LEN': 3,
    'USE_TEST': False,
    'WANDB': False,
    'LABEL_SMOOTHING': 0.0,
    'OPTIMIZER': 'adam',
    'CLEANED_DATASET': True,
    'NUM_RUNS': 1,

    'GRAD_CLIPPING': True,
    'LR_SCHEDULER': False,

    'CL_TASK': 'so',  # so or full
    'DS_TYPE': 'transductive',  # transductive or inductive
    'IND_V': 'v1',  # v1 or v2
    'SWAP': False,
    'TR_RATIO': 1.0,
    'VAL_RATIO': 1.0,
    'RANDPERM': False,

    'PYG_DATA': True,
    'USE_FEATURES': False,
    'WEIGHT_LOSS': False,

    # NodePiece args
    'SUBBATCH': 5000,
    'TOKENIZE': False,
    'NUM_ANCHORS': 500,
    'MAX_PATHS': 20,
    'NEAREST': True,
    'POOLER': 'cat',
    'SAMPLE_RELS': 5,
    'RANDOM_HASHES': 0,
    'MAX_PATH_LEN': 0,
    'USE_DISTANCES': True,
    'T_LAYERS': 2,
    'T_HEADS': 4,
    'T_HIDDEN': 512,
    'T_DROP': 0.1,
    'NO_ANC': False,
}


STAREARGS = {
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 80,
    'GCN_DROP': 0.1,
    'HID_DROP': 0.3,
    'BIAS': False,
    'OPN': 'rotate',
    'TRIPLE_QUAL_WEIGHT': 0.8,
    'QUAL_AGGREGATE': 'sum',  # or concat or mul
    'QUAL_OPN': 'rotate',
    'QUAL_N': 'sum',  # or mean
    'SUBBATCH': 0,
    'QUAL_REPR': 'sparse',  # sparse or full
    'ATTENTION': False,
    'ATTENTION_HEADS': 4,
    'ATTENTION_SLOPE': 0.2,
    'ATTENTION_DROP': 0.1,

    # LRGA
    'LRGA': False,
    'LRGA_K': 50,
    'LRGA_DROP': 0.2

}

DEFAULT_CONFIG['STAREARGS'] = STAREARGS

if __name__ == "__main__":

    # Get parsed arguments
    config = DEFAULT_CONFIG.copy()
    gcnconfig = STAREARGS.copy()
    parsed_args = parse_args(sys.argv[1:])
    print(parsed_args)

    # Superimpose this on default config
    for k, v in parsed_args.items():
        # If its a generic arg
        if k in config.keys():
            default_val = config[k.upper()]
            if default_val is not None:
                needed_type = type(default_val)
                config[k.upper()] = needed_type(v)
            else:
                config[k.upper()] = v
        # If its a gcnarg
        elif k.lower().startswith('gcn_') and k[4:] in gcnconfig:
            default_val = gcnconfig[k[4:].upper()]
            if default_val is not None:
                needed_type = type(default_val)
                gcnconfig[k[4:].upper()] = needed_type(v)
            else:
                gcnconfig[k[4:].upper()] = v
        else:
            config[k.upper()] = v

    config['STAREARGS'] = gcnconfig

    # Load data, by default we are in the Transductive NC mode, predicting labels of subjects/objects nodes ("so" task)
    data = load_clean_pyg(name=config["DATASET"],
                              subtype="triples" if config["STATEMENT_LEN"] == 3 else "statements",
                              task=config["CL_TASK"],
                              inductive=config['DS_TYPE'],
                              ind_v=config['IND_V'],
                              maxlen=config["MAX_QPAIRS"],
                              permute=config['RANDPERM'])
    train_graph, val_graph, test_graph = data['train_graph'], data['val_graph'], data['test_graph']

    # we don't use features, but rather train node embeddings from scratch or based on anchors
    if config['USE_FEATURES']:
        config['FEATURE_DIM'] = train_graph.x.shape[1]


    config['NUM_ENTITIES'] = data["n_entities"]
    config['NUM_RELATIONS'] = data["n_relations"]
    train_mask, val_mask, test_mask = data["train_mask"], data["valid_mask"], data["test_mask"]
    train_y, val_y, test_y = data["train_y"], data["val_y"], data["test_y"]
    all_labels, label2id = data["all_labels"], data["label2id"]

    # Apply NodePiece to create a hash vocab
    if config['TOKENIZE']:
        enriched_graph = train_graph.clone()
        # Add reverse stuff
        reverse_index = torch.zeros_like(enriched_graph.edge_index)
        reverse_index[1, :] = enriched_graph.edge_index[0, :]
        reverse_index[0, :] = enriched_graph.edge_index[1, :]
        rev_edge_type = enriched_graph.edge_type + config['NUM_RELATIONS']

        enriched_graph.edge_index = torch.cat([enriched_graph.edge_index, reverse_index], dim=1)
        enriched_graph.edge_type = torch.cat([enriched_graph.edge_type, rev_edge_type], dim=0)
        kg_tokenizer = NodePiece_Tokenizer(triples=enriched_graph,
                                anchor_strategy={
                                    "degree": 0.4,
                                    "betweenness": 0.0,
                                    "pagerank": 0.4,
                                    "random": 0.2
                                },
                                num_anchors=config['NUM_ANCHORS'], dataset_name=config['DATASET'],
                                limit_shortest=100, add_identity=False, mode="path", limit_random=0)
        if config['MAX_PATH_LEN'] == 0:
            config['MAX_PATH_LEN'] = kg_tokenizer.max_seq_len
    else:
        kg_tokenizer = None
        enriched_graph = None

    # prepare labels
    config['NUM_CLASSES'] = len(all_labels)

    # adjust the train/val size for our semi-supervised NC task
    if config['USE_TEST']:
        input_data = {"train": train_y, "eval": test_y}
    else:
        if config['SWAP']:
            if config['TR_RATIO'] < 1.0:
                tr_keys = random.sample(list(val_y), int(len(val_y)*config['TR_RATIO']))
                val_y = {k: val_y[k] for k in tr_keys}
            if config['VAL_RATIO'] < 1.0:
                vl_keys = random.sample(list(train_y), int(len(train_y)*config['VAL_RATIO']))
                train_y = {k: train_y[k] for k in vl_keys}
            input_data = {"train": val_y, "eval": train_y}
        else:
            if config['TR_RATIO'] < 1.0:
                tr_keys = random.sample(list(train_y), int(len(train_y)*config['TR_RATIO']))
                train_y = {k: train_y[k] for k in tr_keys}
            if config['VAL_RATIO'] < 1.0:
                vl_keys = random.sample(list(val_y), int(len(val_y)*config['VAL_RATIO']))
                val_y = {k: val_y[k] for k in vl_keys}
            input_data = {"train": train_y, "eval": val_y}

    print(f"Training on {len(input_data['train'])} entities")

    config['DEVICE'] = torch.device(config['DEVICE'])

    # create the model
    if config['MODEL_NAME'].lower() == 'stare':
        model = StarE_PyG_NC(config, tokenizer=kg_tokenizer, graph=enriched_graph)
    elif config['MODEL_NAME'].lower() == 'mlp':
        model = MLP_PyG(config, tokenizer=kg_tokenizer, graph=enriched_graph)
    else:
        raise BadParameters(f"Unknown Model Name {config['MODEL_NAME']}")

    print("Model params ", sum([param.nelement() for param in model.parameters()]))
    print(f"Model params  {sum(p.numel() for p in model.parameters())}")

    if config['SAVE']:
        savedir = Path(f"./models/{config['DATASET']}/{config['MODEL_NAME']}")
        if not savedir.exists(): savedir.mkdir(parents=True)
        savedir = mt_save_dir(savedir, _newdir=True)
        save_content = {'model': model, 'config': config}
    else:
        savedir, save_content = None, None

    # Arguments for the training loop
    args = {
        "epochs": config['EPOCHS'],
        "train_fn": model,
        "device": config['DEVICE'],
        "eval_fn": eval_classification,
        "eval_every": config['EVAL_EVERY'],
        "log_wandb": config['WANDB'],
        "run_trn_testbench": config['RUN_TESTBENCH_ON_TRAIN'],
        "savedir": savedir,
        "save_content": save_content,
        "grad_clipping": config['GRAD_CLIPPING'],
        "scheduler": None
    }

    training_loop = training_loop_pyg_nc
    sampler = NodeClSampler(data=input_data,
                            num_labels=len(all_labels),
                            label2id=label2id,
                            lbl_smooth=config['LABEL_SMOOTHING'])

    args['data_fn'] = sampler.get_data
    if config['WEIGHT_LOSS']:
        args['criterion'] = torch.nn.BCEWithLogitsLoss(pos_weight=sampler.pos_weights.to(config['DEVICE']))
    else:
        args['criterion'] = torch.nn.BCEWithLogitsLoss()

    if config['PYG_DATA']:
        args['train_graph'] = train_graph
        if config['DS_TYPE'] == "transductive":
            args['val_graph'] = train_graph  # val and test masks are still on the train graph
        else:
            args['val_graph'] = val_graph if not config['USE_TEST'] else test_graph
        args['model'] = model

    traces = []
    wandb_name = None
    for run in range(config['NUM_RUNS']):
        model.reset_parameters()
        model.to(config['DEVICE'])
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
        if config['LR_SCHEDULER']:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)
            args['scheduler'] = scheduler
        args['opt'] = optimizer

        if config['WANDB']:
            wandb_run = wandb.init(project="NodePiece_NC", reinit=True, settings=wandb.Settings(start_method='fork'))
            wandb_name = wandb.run.name if wandb_name is None else wandb_name
            wandb.run.name = f"{wandb_name}-{str(run)}"
            #wandb.run.save()
            for k, v in config.items():
                wandb.config[k] = v


        trace = training_loop(**args)
        traces.append(trace)

        if config['WANDB']:
            wandb_run.join()

    print_results(traces)



