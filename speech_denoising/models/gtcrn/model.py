from ..base import BaseEnhancementModel


class GTCRNModel(BaseEnhancementModel):
    """
    GTCRN adapter.

    Current stage:
    - interface only
    - no checkpoint
    - no training logic

    The original GTCRN architecture will be integrated later.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.config = kwargs
        self.network = None

    def forward(self, noisy_audio):
        if self.network is None:
            raise RuntimeError(
                "GTCRN network has not been integrated yet."
            )

        return self.network(noisy_audio)
