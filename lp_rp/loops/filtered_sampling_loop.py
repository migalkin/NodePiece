from pykeen.training import SLCWATrainingLoop
# from pykeen105.negative_sampler import SelfAdversarialNegativeSampler, RelationalNegativeSampler
import pdb
class FilteredSLCWATrainingLoop(SLCWATrainingLoop):

    @property
    def num_negs_per_pos_rel(self) -> int:
        """Return number of negatives per positive from the sampler.

        Property for API compatibility
        """
        return self.negative_sampler.num_negs_per_pos_rel

    def _mr_loss_helper(self, positive_scores, negative_scores, _label_smoothing=None):
        if self.num_negs_per_pos + self.num_negs_per_pos_rel > 1:
            positive_scores = positive_scores.repeat(self.num_negs_per_pos + self.num_negs_per_pos_rel, 1)

        return self.model.compute_mr_loss(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        )