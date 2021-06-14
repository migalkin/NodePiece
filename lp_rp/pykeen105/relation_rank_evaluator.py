from pykeen.evaluation import Evaluator, MetricResults, RankBasedEvaluator
from pykeen.models.base import Model, MappedTriples
from pykeen.evaluation.evaluator import create_dense_positive_mask_, filter_scores_, timeit, split_list_in_batches_iter, tqdm, optional_context_manager, logger
from pykeen.evaluation.rank_based_evaluator import pd, RankBasedMetricResults, dataclass, field, dataclass_json, fix_dataclass_init_docs, compute_rank_from_scores, RANK_TYPES, RANK_BEST, RANK_WORST, RANK_AVERAGE, RANK_AVERAGE_ADJUSTED
from collections import defaultdict
import numpy as np
from typing import Any, Collection, List, Mapping, Optional, Tuple, Union, Dict
import torch
import pdb


@fix_dataclass_init_docs
@dataclass_json
@dataclass
class RelationPredictionRankBasedMetricResults(RankBasedMetricResults):
    #: The mean over all ranks: mean_i r_i. Lower is better.
    mean_rank: Dict[str, float] = field(metadata=dict(
        doc='The mean over all ranks: mean_i r_i. Lower is better.',
    ))

    #: The mean over all reciprocal ranks: mean_i (1/r_i). Higher is better.
    mean_reciprocal_rank: Dict[str, float] = field(metadata=dict(
        doc='The mean over all reciprocal ranks: mean_i (1/r_i). Higher is better.',
    ))

    #: The hits at k for different values of k, i.e. the relative frequency of ranks not larger than k.
    #: Higher is better.
    hits_at_k: Dict[str, Dict[int, float]] = field(metadata=dict(
        doc='The hits at k for different values of k, i.e. the relative frequency of ranks not larger than k.'
            ' Higher is better.',
    ))

    #: The mean over all chance-adjusted ranks: mean_i (2r_i / (num_entities+1)). Lower is better.
    #: Described by [berrendorf2020]_.
    adjusted_mean_rank: Dict[str, float] = field(metadata=dict(
        doc='The mean over all chance-adjusted ranks: mean_i (2r_i / (num_entities+1)). Lower is better.',
    ))

    def get_metric(self, name: str) -> float:  # noqa: D102
        dot_count = name.count('.')
        if 0 == dot_count:  # assume average by default
            rank_type, metric = 'avg', name
        elif 1 == dot_count:
            rank_type, metric = name.split('.')
        else:
            raise ValueError(f'Malformed metric name: {name}')

        if rank_type not in RANK_AVERAGE and metric in {'adjusted_mean_rank'}:
            raise ValueError(f'Invalid rank type for adjusted mean rank: {rank_type}. Allowed type: {RANK_AVERAGE}')
        elif rank_type not in RANK_TYPES:
            raise ValueError(f'Invalid rank type: {rank_type}. Allowed types: {RANK_TYPES}')

        if metric in {'mean_rank', 'mean_reciprocal_rank'}:
            return getattr(self, metric)[rank_type]
        elif metric in {'adjusted_mean_rank'}:
            return getattr(self, metric)

        rank_type_hits_at_k = self.hits_at_k[rank_type]
        for prefix in ('hits_at_', 'hits@'):
            if not metric.startswith(prefix):
                continue
            k = metric[len(prefix):]
            k = 10 if k == 'k' else int(k)
            return rank_type_hits_at_k[k]

        raise ValueError(f'Invalid metric name: {name}')

    def to_flat_dict(self):  # noqa: D102
        r = {'avg.adjusted_mean_rank': self.adjusted_mean_rank}
        for rank_type in RANK_TYPES:
            r[f'{rank_type}.mean_rank'] = self.mean_rank[rank_type]
            r[f'{rank_type}.mean_reciprocal_rank'] = self.mean_reciprocal_rank[rank_type]
            for k, v in self.hits_at_k[rank_type].items():
                r[f'{rank_type}.hits_at_{k}'] = v
        return r

    def to_df(self):
        """Output the metrics as a pandas dataframe."""
        rows = [
            ('avg', 'adjusted_mean_rank', self.adjusted_mean_rank)
        ]
        for rank_type in RANK_TYPES:
            rows.append((rank_type, 'mean_rank', self.mean_rank[rank_type]))
            rows.append((rank_type, 'mean_reciprocal_rank', self.mean_reciprocal_rank[rank_type]))
            for k, v in self.hits_at_k[rank_type].items():
                rows.append((rank_type, f'hits_at_{k}', v))
        return pd.DataFrame(rows, columns=['Type', 'Metric', 'Value'])


class RelationPredictionRankBasedEvaluator(Evaluator):
    def __init__(
        self,
        ks = None,
        filtered: bool = True,
    ):
        super().__init__(filtered=filtered)
        self.ks = tuple(ks) if ks is not None else (1, 3, 5, 10)
        for k in self.ks:
            if isinstance(k, float) and not (0 < k < 1):
                raise ValueError(
                    'If k is a float, it should represent a relative rank, i.e. a value between 0 and 1 (excl.)',
                )
        self.ranks = defaultdict(list)
        self.num_relations = None

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:
        pass

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:
        pass

    def _update_ranks_(self, true_scores, all_scores):
        batch_ranks = compute_rank_from_scores(
            true_score=true_scores,
            all_scores=all_scores,
        )
        self.num_relations = all_scores.shape[1]
        for k, v in batch_ranks.items():
            self.ranks[k].extend(v.detach().cpu().tolist())

    def evaluate(
            self, model, mapped_triples = None, batch_size = None, slice_size = None, device = None,
            use_tqdm = True, tqdm_kwargs = None, restrict_entities_to = None, do_time_consuming_checks = True,
            additional_filter_triples=None,
    ):
        if mapped_triples is None:
            mapped_triples = model.triples_factory.mapped_triples

        if batch_size is None and model.automatic_memory_optimization:
            batch_size, slice_size = self.batch_and_slice(
                model=model,
                mapped_triples=mapped_triples,
                batch_size=batch_size,
                device=device,
                use_tqdm=False,
                restrict_entities_to=restrict_entities_to,
                do_time_consuming_checks=do_time_consuming_checks,
            )
            # The batch_size and slice_size should be accessible to outside objects for re-use, e.g. early stoppers.
            self.batch_size = batch_size
            self.slice_size = slice_size

            # Clear the ranks from the current evaluator
            self.finalize()

        return evaluate(
            model=model,
            mapped_triples=mapped_triples,
            additional_filtered_triples=additional_filter_triples,
            evaluators=self,
            batch_size=batch_size,
            slice_size=slice_size,
            device=device,
            squeeze=True,
            use_tqdm=use_tqdm,
            tqdm_kwargs=tqdm_kwargs,
            restrict_relations_to=restrict_entities_to,
            do_time_consuming_checks=do_time_consuming_checks,
        )

    def _get_ranks(self, rank_type):
        return np.asarray(self.ranks.get(rank_type, []), dtype=np.float64)

    def finalize(self) -> MetricResults:
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at_k = {}
        adjusted_mean_rank = {}

        for rank_type in RANK_TYPES:
            ranks = self._get_ranks(rank_type=rank_type)
            if len(ranks) < 1:
                continue
            hits_at_k[rank_type] = {
                k: np.mean(ranks <= k) if isinstance(k, int) else np.mean(ranks <= int(self.num_relations * k))
                for k in self.ks
            }
            mean_rank[rank_type] = np.mean(ranks)
            mean_reciprocal_rank[rank_type] = np.mean(np.reciprocal(ranks))

        adjusted_ranks = self._get_ranks(rank_type=RANK_AVERAGE_ADJUSTED)
        if len(adjusted_ranks) >= 1:
            adjusted_mean_rank = float(np.mean(adjusted_ranks))

        self.ranks.clear()

        return RelationPredictionRankBasedMetricResults(
            mean_rank=dict(mean_rank),
            mean_reciprocal_rank=dict(mean_reciprocal_rank),
            hits_at_k=dict(hits_at_k),
            adjusted_mean_rank=adjusted_mean_rank,
        )

    def process_relation_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores)


def get_unique_relation_ids_from_triples_tensor(mapped_triples: MappedTriples) -> torch.LongTensor:
    """Return the unique entity IDs used in a tensor of triples."""
    return mapped_triples[:, 1].unique()

def evaluate(
    model: Model,
    mapped_triples: MappedTriples,
    evaluators: Union[Evaluator, Collection[Evaluator]],
    additional_filtered_triples: Union[None, MappedTriples, List[MappedTriples]] = None,
    only_size_probing: bool = False,
    batch_size: Optional[int] = None,
    slice_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    squeeze: bool = True,
    use_tqdm: bool = True,
    tqdm_kwargs: Optional[Mapping[str, str]] = None,
    restrict_relations_to: Optional[torch.LongTensor] = None,
    do_time_consuming_checks: bool = True,
) -> Union[MetricResults, List[MetricResults]]:
    """Evaluate metrics for model on mapped triples.

    The model is used to predict scores for all tails and all heads for each triple. Subsequently, each abstract
    evaluator is applied to the scores, also receiving the batch itself (e.g. to compute entity-specific metrics).
    Thereby, the (potentially) expensive score computation against all entities is done only once. The metric evaluators
    are expected to maintain their own internal buffers. They are returned after running the evaluation, and should
    offer a possibility to extract some final metrics.

    :param model:
        The model to evaluate.
    :param mapped_triples:
        The triples on which to evaluate.
    :param evaluators:
        An evaluator or a list of evaluators working on batches of triples and corresponding scores.
    :param only_size_probing:
        The evaluation is only performed for two batches to test the memory footprint, especially on GPUs.
    :param batch_size: >0
        A positive integer used as batch size. Generally chosen as large as possible. Defaults to 1 if None.
    :param slice_size: >0
        The divisor for the scoring function when using slicing.
    :param device:
        The device on which the evaluation shall be run. If None is given, use the model's device.
    :param squeeze:
        Return a single instance of :class:`MetricResults` if only one evaluator was given.
    :param use_tqdm:
        Should a progress bar be displayed?
    :param restrict_relations_to:
        Optionally restrict the evaluation to the given relations IDs. This may be useful if one is only interested in a
        part of the relations, e.g. due to type constraints, but wants to train on all available data. For ranking the
        entities, we still compute all scores for all possible replacement entities to avoid irregular access patterns
        which might decrease performance, but the scores with afterwards be filtered to only keep those of interest.
        If provided, we assume that the triples are already filtered, such that it only contains the entities of
        interest.
    :param do_time_consuming_checks:
        Whether to perform some time consuming checks on the provided arguments. Currently, this encompasses:
        - If restrict_entities_to is not None, check whether the triples have been filtered.
        Disabling this option can accelerate the method.
    """
    if isinstance(evaluators, Evaluator):  # upgrade a single evaluator to a list
        evaluators = [evaluators]

    start = timeit.default_timer()

    # verify that the triples have been filtered
    if restrict_relations_to is not None and do_time_consuming_checks:
        present_relation_ids = set(get_unique_relation_ids_from_triples_tensor(mapped_triples=mapped_triples).tolist())
        unwanted = present_relation_ids.difference(restrict_relations_to.tolist())
        if len(unwanted) > 0:
            raise ValueError(f'mapped_triples contains IDs of entities which are not contained in restrict_relations_to:'
                             f'{unwanted}. This will invalidate the evaluation results.')

    # Send to device
    if device is not None:
        model = model.to(device)
    device = model.device

    # Ensure evaluation mode
    model.eval()

    # Split evaluators into those which need unfiltered results, and those which require filtered ones
    filtered_evaluators = list(filter(lambda e: e.filtered, evaluators))
    unfiltered_evaluators = list(filter(lambda e: not e.filtered, evaluators))

    # Check whether we need to be prepared for filtering
    filtering_necessary = len(filtered_evaluators) > 0

    # Check whether an evaluator needs access to the masks
    # This can only be an unfiltered evaluator.
    positive_masks_required = any(e.requires_positive_mask for e in unfiltered_evaluators)

    # Prepare for result filtering
    if filtering_necessary or positive_masks_required:
        #all_pos_triples = torch.cat([model.triples_factory.mapped_triples, mapped_triples], dim=0)
        if additional_filtered_triples is None:
            logger.warning("The filtered setting was enabled, but there were no `additional_filtered_triples"
                           "given. This means you probably forgot to pass (at least) the training triples. Try:"
                           "additional_filtered_triples=[dataset.training.mapped_triples]"
                           "Or if you want to use the Bordes et al. (2013) approach to filtering, do:"
                           "additional_filtered_triples=[dataset.training.mapped_triples,dataset.validation.mapped_triples,]")
            all_pos_triples = mapped_triples
        elif isinstance(additional_filtered_triples, (list, tuple)):
            all_pos_triples = torch.cat([*additional_filtered_triples, mapped_triples], dim=0)
        else:
            all_pos_triples = torch.cat([additional_filtered_triples, mapped_triples], dim=0)
        all_pos_triples = all_pos_triples.to(device=device)
    else:
        all_pos_triples = None

    # Send tensors to device
    mapped_triples = mapped_triples.to(device=device)

    # Prepare batches
    if batch_size is None:
        batch_size = 1
    batches = split_list_in_batches_iter(input_list=mapped_triples, batch_size=batch_size)

    # Show progressbar
    num_triples = mapped_triples.shape[0]

    # Flag to check when to quit the size probing
    evaluated_once = False

    # Disable gradient tracking
    _tqdm_kwargs = dict(
        desc=f'Evaluating on {model.device}',
        total=num_triples,
        unit='triple',
        unit_scale=True,
        # Choosing no progress bar (use_tqdm=False) would still show the initial progress bar without disable=True
        disable=not use_tqdm,
    )
    if tqdm_kwargs:
        _tqdm_kwargs.update(tqdm_kwargs)
    with optional_context_manager(use_tqdm, tqdm(**_tqdm_kwargs)) as progress_bar, torch.no_grad():
        # batch-wise processing
        for batch in batches:
            batch_size = batch.shape[0]
            _evaluate_batch(
                batch=batch,
                model=model,
                filtered_evaluators=filtered_evaluators,
                unfiltered_evaluators=unfiltered_evaluators,
                slice_size=slice_size,
                all_pos_triples=all_pos_triples,
                restrict_relations_to=restrict_relations_to,
                positive_masks_required=positive_masks_required,
                filtering_necessary=filtering_necessary,
            )

            # If we only probe sizes we do not need more than one batch
            if only_size_probing and evaluated_once:
                break

            evaluated_once = True

            if use_tqdm:
                progress_bar.update(batch_size)

        # Finalize
        results = [evaluator.finalize() for evaluator in evaluators]

    stop = timeit.default_timer()
    if only_size_probing:
        logger.debug("Evaluation took %.2fs seconds", stop - start)
    else:
        logger.info("Evaluation took %.2fs seconds", stop - start)

    if squeeze and len(results) == 1:
        return results[0]

    return results



def _evaluate_batch(
    batch,
    model,
    filtered_evaluators: Collection[Evaluator],
    unfiltered_evaluators: Collection[Evaluator],
    slice_size: Optional[int],
    all_pos_triples: Optional[MappedTriples],
    restrict_relations_to: Optional[torch.LongTensor],
    positive_masks_required: bool,
    filtering_necessary: bool,
) -> torch.BoolTensor:
    """
    Evaluate batch for all head predictions(column=0), or all tail predictions (column=2).

    :param batch: shape: (batch_size, 3)
        The batch of currently evaluated triples.
    :param model:
        The model to evaluate.
    :param column:
        The column which to evaluate. Either 0 for head prediction, or 2 for tail prediction.
    :param filtered_evaluators:
        The evaluators which work on filtered scores.
    :param unfiltered_evaluators:
        The evaluators which work on unfiltered scores.
    :param slice_size:
        An optional slice size for computing the scores.
    :param all_pos_triples:
        All positive triples (required if filtering is necessary).
    :param restrict_relations_to:
        Restriction to evaluate only for these relations.
    :param positive_masks_required:
        Whether dense positive masks are required (by any unfiltered evaluator).
    :param filtering_necessary:
        Whether filtering is necessary.

    :return:
        The relation filter, which can be re-used for the same batch.
    """

    # Predict scores once
    batch_scores_of_corrupted = model.predict_scores_all_relations(batch[:, [0, 2]], slice_size=slice_size)

    # Select scores of true
    batch_scores_of_true = batch_scores_of_corrupted[
        torch.arange(0, batch.shape[0]),
        batch[:, 1],
    ]

    # Create positive filter for all corrupted
    if filtering_necessary or positive_masks_required:
        # Needs all positive triples
        if all_pos_triples is None:
            raise ValueError('If filtering_necessary of positive_masks_required is True, all_pos_triples has to be '
                             'provided, but is None.')

        # Create filter
        positive_filter = create_sparse_positive_filter_(
            hrt_batch=batch,
            all_pos_triples=all_pos_triples,
        )

    # Create a positive mask with the size of the scores from the positive filter
    if positive_masks_required:
        positive_mask = create_dense_positive_mask_(
            zero_tensor=torch.zeros_like(batch_scores_of_corrupted),
            filter_batch=positive_filter,
        )
    else:
        positive_mask = None

    # Restrict to entities of interest
    if restrict_relations_to is not None:
        batch_scores_of_corrupted_ = batch_scores_of_corrupted[:, restrict_relations_to]
        positive_mask = positive_mask[:, restrict_relations_to]
    else:
        batch_scores_of_corrupted_ = batch_scores_of_corrupted

    # Evaluate metrics on these *unfiltered* scores
    for unfiltered_evaluator in unfiltered_evaluators:
        unfiltered_evaluator.process_relation_scores_(
            hrt_batch=batch,
            true_scores=batch_scores_of_true[:, None],
            scores=batch_scores_of_corrupted_,
            dense_positive_mask=positive_mask,
        )

    # Filter
    if filtering_necessary:
        batch_filtered_scores_of_corrupted = filter_scores_(
            scores=batch_scores_of_corrupted,
            filter_batch=positive_filter,
        )

        # The scores for the true triples have to be rewritten to the scores tensor
        batch_filtered_scores_of_corrupted[
            torch.arange(0, batch.shape[0]),
            batch[:, 1],
        ] = batch_scores_of_true

        # Restrict to entities of interest
        if restrict_relations_to is not None:
            batch_filtered_scores_of_corrupted = batch_filtered_scores_of_corrupted[:, restrict_relations_to]

        # Evaluate metrics on these *filtered* scores
        for filtered_evaluator in filtered_evaluators:
            filtered_evaluator.process_relation_scores_(
                hrt_batch=batch,
                true_scores=batch_scores_of_true[:, None],
                scores=batch_filtered_scores_of_corrupted,
            )

def create_sparse_positive_filter_(
    hrt_batch: MappedTriples,
    all_pos_triples: torch.LongTensor
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    # Split batch
    batch_heads, batch_tails = hrt_batch[:, 0:1], hrt_batch[:, 2:3]
    head_filter = (all_pos_triples[:, 0]).view(1, -1) == batch_heads
    tail_filter = (all_pos_triples[:, 2]).view(1, -1) == batch_tails
    filter_batch = (head_filter & tail_filter).nonzero(as_tuple=False)
    filter_batch[:, 1] = all_pos_triples[:, 1].view(1, -1)[:, filter_batch[:, 1]]

    return filter_batch
