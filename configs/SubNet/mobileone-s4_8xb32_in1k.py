_base_ = [
    '../_base_/models/mobileone/mobileone_s4.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr_coswd_300e.py',
    '../_base_/default_runtime.py'
]

# schedule settings，权重衰减 0.0
optim_wrapper = dict(paramwise_cfg=dict(norm_decay_mult=0.))

# dataloader， batch size
val_dataloader = dict(batch_size=256)
test_dataloader = dict(batch_size=256)


bgr_mean = _base_.data_preprocessor['mean'][::-1]

# 取出 base_train_pipeline
base_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),          # 随机翻转
    dict(
        type='RandAugment',                                             # 随机增强
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(pad_val=[round(x) for x in bgr_mean])),
    dict(type='PackInputs')                                             # 打包数据
]

import copy  # noqa: E402

# 对流水线中的参数进行修改， 在 1epoch之后，图像大小变为160， 137epoch之后图像大小变为192， 112epoch之后，图像大小变为224
# modify start epoch RandomResizedCrop.scale to 160
# and RA.magnitude_level * 0.3
train_pipeline_1e = copy.deepcopy(base_train_pipeline)
train_pipeline_1e[1]['scale'] = 160
train_pipeline_1e[3]['magnitude_level'] *= 0.3
_base_.train_dataloader.dataset.pipeline = train_pipeline_1e

# modify 137 epoch's RandomResizedCrop.scale to 192
# and RA.magnitude_level * 0.7
train_pipeline_37e = copy.deepcopy(base_train_pipeline)
train_pipeline_37e[1]['scale'] = 192
train_pipeline_37e[3]['magnitude_level'] *= 0.7

# modify 112 epoch's RandomResizedCrop.scale to 224
# and RA.magnitude_level * 1.0
train_pipeline_112e = copy.deepcopy(base_train_pipeline)
train_pipeline_112e[1]['scale'] = 224
train_pipeline_112e[3]['magnitude_level'] *= 1.0

# 自定义的 hook 有 EMA，切换流水线的 hook 是 SwitchRecipeHook
custom_hooks = [
    dict(
        type='SwitchRecipeHook',
        schedule=[
            dict(action_epoch=37, pipeline=train_pipeline_37e),
            dict(action_epoch=112, pipeline=train_pipeline_112e),
        ]),
    dict(
        type='EMAHook',
        momentum=5e-4,
        priority='ABOVE_NORMAL',
        update_buffers=True)
]
