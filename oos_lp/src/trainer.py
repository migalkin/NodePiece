# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn

from model.DisMultOutKG import DisMultOutKG
from model.dm_tokenized import TokenizedDistMult
from utils import save_model
from tqdm import tqdm
from tester import OutKGTester

from pykeen.losses import NSSALoss
import wandb


class OutKGTrainer:
    def __init__(self, dataset, args, tokenizer):
        self.dataset = dataset
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.args.tokenize:
            self.model = TokenizedDistMult(self.args, self.device, dataset, tokenizer)
        else:
            self.model = DisMultOutKG(self.dataset, self.args, self.device)
        #self.model = nn.DataParallel(self.model)

        if self.args.loss_fc == "spl":
            self.predict_loss = nn.Softplus()
        else:
            self.predict_loss = NSSALoss(margin=args.margin)

        if self.args.wandb:
            wandb.init(project="oog_token", entity='lilbert', reinit=True, settings=wandb.Settings(start_method='fork'))
            wandb.config.update(vars(self.args))

        #self.tester = OutKGTester(dataset)


    def l2_loss(self):
        return self.model.l2_loss()  # removed model.module

    def train(self, save=True):
        self.model.to(self.device)  # for bypassing torch DataParallel
        self.model.train()
        if self.args.use_acc:
            initial_accumulator_value = 0.1
        else:
            initial_accumulator_value = 0.0

        if self.args.use_custom_reg:
            weight_decay = 0.0
        else:
            weight_decay = self.args.reg_lambda

        if self.args.opt == "adagrad":
            print("using adagrad")
            optimizer = torch.optim.Adagrad(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                # this is added because of the consistency to the original tensorflow code
            )

        else:
            print("using adam")
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.reg_lambda,
            )
        print(f"Number of params: {sum(p.numel() for p in self.model.parameters())}")
        iters_per_update = self.args.simulated_batch_size // self.args.batch_size

        if iters_per_update < 1:
            raise ("Actual batch size smaller than batch size to be simulated.")
        else:
            print("iterations before the gradient step : ", iters_per_update)

        for epoch in tqdm(range(self.args.ne)):
            optimizer.zero_grad()
            last_batch = False
            total_loss = 0.0
            pbar = tqdm(total=self.dataset.num_batch(self.args.batch_size))
            num_iters = 1
            while not last_batch:

                triples, l, new_ent_mask = self.dataset.next_batch(
                    self.args.batch_size,
                    neg_ratio=self.args.neg_ratio,
                    device=self.device,
                )
                last_batch = self.dataset.was_last_batch()

                if self.args.loss_fc == "spl":
                    scores, predicted_emb = self.model(triples, new_ent_mask)
                    predict_loss = torch.sum(self.predict_loss(-l * scores))
                else:
                    pos_batch = triples[:self.args.batch_size, ...]
                    neg_batch = triples[self.args.batch_size:, ...]
                    positive_scores, _ = self.model(pos_batch, None)
                    negative_scores, _ = self.model(neg_batch, None)
                    if self.args.neg_ratio > 1:
                        positive_scores = positive_scores.repeat_interleave(self.args.neg_ratio)
                    predict_loss = self.predict_loss(positive_scores, negative_scores)
                if self.args.use_custom_reg:
                    if num_iters % iters_per_update == 0 or last_batch == True:
                        l2_loss = (
                                self.args.reg_lambda
                                * self.l2_loss()
                                / (
                                    self.dataset.num_batch_simulated(
                                        self.args.simulated_batch_size
                                    )
                                )
                        )
                        loss = predict_loss + l2_loss

                    else:
                        loss = predict_loss
                else:
                    loss = predict_loss

                loss.backward()
                if num_iters % iters_per_update == 0 or last_batch == True:
                    if last_batch:
                        print("last batch triggered gradient update.")
                        print(
                            "remaining iters for gradient update :",
                            num_iters % iters_per_update,
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += loss
                #print(num_iters)
                num_iters += 1
                pbar.update(1)


            print(
                "Loss in iteration "
                + str(epoch)
                + ": "
                + str(total_loss.item() / num_iters)
                + "("
                + self.dataset.dataset_name
                + ")"
            )

            if self.args.eval_every > 0 and epoch % self.args.eval_every == 0:
                tester = OutKGTester(self.dataset)
                with torch.no_grad():
                    mrr = tester.test(self.model, valid_or_test="valid")
                mrr, mr, hits_1, hits_3, hits_10 = tester.measure.mrr, tester.measure.mr, tester.measure.hit1, tester.measure.hit3, tester.measure.hit10

                wandb_log_dict = {
                    'step': epoch,
                    'loss': float(total_loss.item() / num_iters),
                    'val.mrr': mrr.item(),
                    'val.mr': mr.item(),
                    'val.hits_1': hits_1,
                    'val.hits_3': hits_3,
                    'val.hits_10': hits_10
                }
            else:
                wandb_log_dict = {'step': epoch, 'loss': float(total_loss.item() / num_iters)}

            # log to wandb
            if self.args.wandb:
                wandb.log(wandb_log_dict)


            if epoch > 0 and epoch % self.args.save_each == 0 and save:
                print('save model...')
                save_model(
                    self.model,
                    self.args.model_name,
                    self.args.emb_method,
                    self.dataset.dataset_name,
                    epoch,
                    self.args.lr,
                    self.args.reg_lambda,
                    self.args.neg_ratio,
                    self.args.emb_dim,
                )

        print("===== TEST ======")
        test_tester = OutKGTester(self.dataset)
        with torch.no_grad():
            mrr = test_tester.test(self.model, valid_or_test="test")
        mrr, mr, hits_1, hits_3, hits_10 = test_tester.measure.mrr, test_tester.measure.mr, test_tester.measure.hit1, test_tester.measure.hit3, test_tester.measure.hit10

        if self.args.wandb:
            wandb_log_dict = {
                'loss': float(total_loss.item() / num_iters),
                'test.mrr': mrr.item(),
                'test.mr': mr.item(),
                'test.hits_1': hits_1,
                'test.hits_3': hits_3,
                'test.hits_10': hits_10
            }
            wandb.log(wandb_log_dict)
