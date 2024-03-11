_base_ = [
    '../_base_/models/mobiletrack/mobiletrack_s0.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256_coslr_coswd_300e.py',
    '../_base_/default_runtime.py'
]

# schedule settings， 关闭权重衰减
optim_wrapper = dict(paramwise_cfg=dict(norm_decay_mult=0.))

# 设置 batch size
val_dataloader = dict(batch_size=256)
test_dataloader = dict(batch_size=256)

# 自定义 EMA
custom_hooks = [
    dict(
        type='EMAHook',
        momentum=5e-4,
        priority='ABOVE_NORMAL',
        update_buffers=True),
    dict(type='SuperNetHook')
]
