import torch

from torch import nn
from typing import Optional, Iterable, Union, Dict, Tuple, List
from collections import defaultdict
from pykeen.evaluation import RankBasedEvaluator
from pykeen.evaluation.rank_based_evaluator import compute_rank_from_scores

class ILPRankBasedEvaluator(RankBasedEvaluator):

    def __init__(
        self,
        ks: Optional[Iterable[Union[int, float]]] = None,
        filtered: bool = True,
        head_samples: torch.Tensor = None,  # shape: [num_valid_triples, n]
        tail_samples: torch.Tensor = None,  # shape: [num_valid_triples, n]
    ):
        """Initialize rank-based evaluator.

        :param ks:
            The values for which to calculate hits@k. Defaults to {1,3,5,10}.
        :param filtered:
            Whether to use the filtered evaluation protocol. If enabled, ranking another true triple higher than the
            currently considered one will not decrease the score.
        """
        super().__init__(filtered=filtered)
        self.ks = tuple(ks) if ks is not None else (1, 3, 5, 10)
        for k in self.ks:
            if isinstance(k, float) and not (0 < k < 1):
                raise ValueError(
                    'If k is a float, it should represent a relative rank, i.e. a value between 0 and 1 (excl.)',
                )
        self.ranks: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.num_entities = None

        self.head_samples = head_samples
        self.tail_samples = tail_samples


    def _update_ranks_(
        self,
        true_scores: torch.FloatTensor,
        all_scores: torch.FloatTensor,
        side: str,
    ) -> None:
        """Shared code for updating the stored ranks for head/tail scores.

        :param true_scores: shape: (batch_size,)
        :param all_scores: shape: (batch_size, num_entities)
        """
        sampled_entities = self.head_samples if side == "head" else self.tail_samples
        batch_ranks = compute_rank_from_scores(
            true_score=true_scores,
            all_scores=all_scores.gather(1, sampled_entities.to(all_scores.device)),
        )
        self.num_entities = all_scores.shape[1]
        for k, v in batch_ranks.items():
            self.ranks[side, k].extend(v.detach().cpu().tolist())