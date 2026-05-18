"""
resistancemap.mortfm.single_cell
=================================

State-aware single-cell objectives for Block C. The v16 self-supervised
reconstruction got stage classifier CV macro-F1 = 0.147 (below the 0.25
random baseline). v17 adds:

  * supervised contrastive (SupCon) over disease stage labels
  * stage-classification auxiliary loss
  * pseudotime-ordinal regularisation
  * patient-batch balancing in the sampler

Every loss is a self-contained torch.nn.Module; the trainer composes
them via :class:`resistancemap.training.loss_router.LossRouter`.
"""

from resistancemap.mortfm.single_cell.state_contrastive_loss import (  # noqa: F401
    supcon_loss,
)
from resistancemap.mortfm.single_cell.pseudotime_regularizer import (  # noqa: F401
    pseudotime_ordinal_loss,
)
from resistancemap.mortfm.single_cell.disease_stage_sampler import (  # noqa: F401
    StratifiedStageSampler,
)
