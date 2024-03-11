from mmengine.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class SuperNetHook(Hook):
    def __init__(self):
        self.epoch_num = 0

    def before_train_epoch(self, runner):
        # runner.model.backbone._get_path_back()
        self.epoch_num = self.epoch_num + 1

        if self.epoch_num <= 10:
            runner.model.backbone.architecture = [[0], [0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0]]

        elif 10 < self.epoch_num <= 20:
            runner.model.backbone.architecture = [[1], [1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1]]

        elif 20 < self.epoch_num <= 30:
            runner.model.backbone.architecture = [[2], [2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                                                  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2]]
        else:
            runner.model.backbone._get_path_back()

        print(runner.model.backbone.architecture)
