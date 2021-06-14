# Copyright (C) 2019 Bahare Fatemi
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# https://github.com/baharefatemi/SimplE
#
class Measure:
    def __init__(self):
        self.hit1 = 0.0
        self.hit3 = 0.0
        self.hit10 = 0.0
        self.mrr = 0.0
        self.mr = 0.0
        self.num_facts = 0.0

    def update(self, rank):
        if rank == 1:
            self.hit1 += 1.0
        if rank <= 3:
            self.hit3 += 1.0
        if rank <= 10:
            self.hit10 += 1.0
        self.mr += rank
        self.mrr += 1.0 / rank
        self.num_facts += 1.0

    def normalize(self):
        if self.hit1 > 0.0:
            self.hit1 /= self.num_facts
        if self.hit3 > 0.0:
            self.hit3 /= self.num_facts
        if self.hit10 > 0.0:
            self.hit10 /= self.num_facts
        self.mr /= self.num_facts
        self.mrr /= self.num_facts

    def print_(self):
        print("\tHit@1 =", self.hit1)
        print("\tHit@3 =", self.hit3)
        print("\tHit@10 =", self.hit10)
        print("\tMR =", self.mr)
        print("\tMRR =", self.mrr)
        print("")
