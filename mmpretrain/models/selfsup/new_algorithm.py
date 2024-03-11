from mmpretrain.registry import MODELS
from .base import BaseSelfSupervisor


@MODELS.register_module()
class NewAlgorithm(BaseSelfSupervisor):

    def __init__(self, backbone, neck=None, head=None, init_cfg=None):
        super().__init__(init_cfg)
        pass

    # ``extract_feat`` function is defined in BaseSelfSupvisor, you could
    # overwrite it if needed
    def extract_feat(self, inputs, **kwargs):
        pass

    # the core function to compute the loss
    def loss(self, inputs, data_samples, **kwargs):
        pass
