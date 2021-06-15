import wandb
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from typing import Callable, Dict, Union
from loops.sampler import NodeClSampler

from utils.utils_mytorch import *
from torch_geometric.data import Data


def training_loop_pyg_nc(epochs: int,
                      opt: torch.optim,
                      model: Callable,
                      train_graph: Data,
                      val_graph: Data,
                      device: torch.device = torch.device('cpu'),
                      data_fn: Callable = NodeClSampler,
                      eval_fn: Callable = None,
                      eval_every: int = 1,
                      log_wandb: bool = True,
                      run_trn_testbench: bool = True,
                      savedir: str = None,
                      save_content: Dict[str, list] = None,
                      grad_clipping: bool = True,
                      scheduler: Callable = None,
                      criterion: Callable = None,
                      **kwargs) -> (list, list, list):
    train_loss = []
    train_rocauc, train_prcauc, train_ap, train_hard_acc = [], [], [], []
    valid_rocauc, valid_prcauc, valid_ap, valid_hard_acc = [], [], [], []


    # Epoch level
    for e in tqdm(range(epochs)):

        # Train
        with Timer() as timer:

            # Get masks and labels
            train_mask, train_y, val_mask, val_y = data_fn()
            model.train()

            opt.zero_grad()

            train_mask_ = torch.tensor(train_mask, dtype=torch.long, device=device)
            train_y_ = torch.tensor(train_y, dtype=torch.float, device=device)
            val_mask_ = torch.tensor(val_mask, dtype=torch.long, device=device)
            val_y_ = torch.tensor(val_y, dtype=torch.float, device=device)

            pred = model(train_graph.to(device=device), train_mask_)

            loss = criterion(pred, train_y_)

            per_epoch_loss = loss.item()

            loss.backward()

            if grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if hasattr(model, "post_parameter_update"):
                model.post_parameter_update()

        # Log this stuff
        print(f"[Epoch: {e} ] Loss: {per_epoch_loss}")
        train_loss.append(per_epoch_loss)

        if e % eval_every == 0 and e >= 1:
            with torch.no_grad():
                model.eval()
                val_preds = torch.sigmoid(model(val_graph.to(device=device), val_mask_))
                val_res = eval_fn(val_y_, val_preds)
                valid_rocauc.append(val_res["rocauc"])
                valid_prcauc.append(val_res["prcauc"])
                valid_ap.append(val_res["ap"])
                valid_hard_acc.append(val_res["hard_acc"])

                if run_trn_testbench:
                    # Also run train testbench
                    train_preds = torch.sigmoid(model(train_graph.to(device=device), train_mask_))
                    unsmoothed_labels = (train_y_ > 0.5).float()
                    tr_res = eval_fn(unsmoothed_labels, train_preds)
                    train_rocauc.append(tr_res["rocauc"])
                    train_prcauc.append(tr_res["prcauc"])
                    train_ap.append(tr_res["ap"])
                    train_hard_acc.append(tr_res["hard_acc"])

                    # Print statement here
                    print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_rocauc: %(tr_rocauc)0.5f | "
                          "Tr_prcauc: %(tr_prcauc)0.5f | Tr_AP: %(tr_ap)0.5f | Tr_hard_acc: %(tr_hard_acc)0.5f |"
                          "Vl_rocauc: %(val_rocauc)0.5f | Vl_prcauc: %(val_prcauc)0.5f | Vl_AP: %(val_ap)0.5f | "
                          "Vl_hard_acc: %(val_hard_acc)0.5f | Time_trn: %(time).3f min"
                          % {'epo': e,
                             'loss': float(per_epoch_loss),
                             'tr_rocauc': float(tr_res["rocauc"]),
                             'tr_prcauc': float(tr_res["prcauc"]),
                             'tr_ap': float(tr_res["ap"]),
                             'tr_hard_acc': float(tr_res["hard_acc"]),
                             'val_rocauc': float(val_res["rocauc"]),
                             'val_prcauc': float(val_res["prcauc"]),
                             'val_ap': float(val_res["ap"]),
                             'val_hard_acc': float(val_res["hard_acc"]),
                             'time': timer.interval / 60.0})

                    if log_wandb:
                        # Wandb stuff
                        wandb.log({
                            'epoch': e,
                            'loss': float(np.mean(per_epoch_loss)),
                            'tr_rocauc': float(tr_res["rocauc"]),
                            'tr_prcauc': float(tr_res["prcauc"]),
                            'tr_ap': float(tr_res["ap"]),
                            'tr_hard_acc': float(tr_res["hard_acc"]),
                            'val_rocauc': float(val_res["rocauc"]),
                            'val_prcauc': float(val_res["prcauc"]),
                            'val_ap': float(val_res["ap"]),
                            'val_hard_acc': float(val_res["hard_acc"]),
                        })

                else:
                    # Don't benchmark over train
                    # Print Statement here
                    print("Epoch: %(epo)03d | Loss: %(loss).5f | "
                          "Vl_rocauc: %(val_rocauc)0.5f | Vl_prcauc: %(val_prcauc)0.5f | Vl_AP: %(val_ap)0.5f | "
                          "Vl_hard_acc: %(val_hard_acc)0.5f | time_trn: %(time).3f min"
                          % {'epo': e,
                             'loss': float(per_epoch_loss),
                             'val_rocauc': float(val_res["rocauc"]),
                             'val_prcauc': float(val_res["prcauc"]),
                             'val_ap': float(val_res["ap"]),
                             'val_hard_acc': float(val_res["hard_acc"]),
                             'time': timer.interval / 60.0})

                    if log_wandb:
                        # Wandb stuff
                        wandb.log({
                            'epoch': e,
                            'loss': float(per_epoch_loss),
                            'val_rocauc': float(val_res["rocauc"]),
                            'val_prcauc': float(val_res["prcauc"]),
                            'val_ap': float(val_res["ap"]),
                            'val_hard_acc': float(val_res["hard_acc"]),
                        })

                # We might wanna save the model, too
                if savedir is not None:
                    mt_save(
                        savedir,
                        torch_stuff=[tosave(obj=save_content['model'].state_dict(), fname='model.torch')],
                        pickle_stuff=[tosave(fname='traces.pkl',
                                             obj=[train_loss, valid_rocauc])],
                        json_stuff=[tosave(obj=save_content['config'], fname='config.json')])
        else:
            # No test benches this time around
            print("Epoch: %(epo)03d | Loss: %(loss).5f |  "
                  "Time_Train: %(time).3f min"
                  % {'epo': e,
                     'loss': float(per_epoch_loss),
                     # 'tracc': float(np.mean(per_epoch_tr_acc)),
                     'time': timer.interval / 60.0})

            if log_wandb:
                # Wandb stuff
                wandb.log({
                    'epoch': e,
                    'loss': float(per_epoch_loss),
                    # 'trn_acc': float(np.mean(per_epoch_tr_acc))
                })

        if scheduler is not None:
            scheduler.step()

    return {
        "loss": train_loss,
        "train_rocauc": train_rocauc,
        "train_prcauc": train_prcauc,
        "train_ap": train_ap,
        "train_hard_acc": train_hard_acc,
        "valid_rocauc": valid_rocauc,
        "valid_prcauc": valid_prcauc,
        "valid_ap": valid_ap,
        "valid_hard_acc": valid_hard_acc
    }
