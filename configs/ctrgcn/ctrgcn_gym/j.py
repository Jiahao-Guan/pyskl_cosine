model = dict(
    type='RecognizerGCN',
    backbone=dict(
		type='CTRGCN',
        graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=99, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/gym/gym_hrnet.pkl'

# left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
# right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
    dict(type='UniformSample', clip_len=50),
    dict(type='PoseDecode'),
    #————————————————————augmentations————————————————————
    # dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    # dict(type='Resize', scale=(-1, 64)),
    # dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    # dict(type='Resize', scale=(56, 56), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    #————————————————————augmentations————————————————————
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
    dict(type='UniformSample', clip_len=50, num_clips=1),
    dict(type='PoseDecode'),
    #————————————————————augmentations————————————————————
    # dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    # dict(type='Resize', scale=(64, 64), keep_ratio=False),
    #————————————————————augmentations————————————————————
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
    dict(type='UniformSample', clip_len=50, num_clips=10),
    dict(type='PoseDecode'),
    #————————————————————augmentations————————————————————
    # dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    # dict(type='Resize', scale=(64, 64), keep_ratio=False),
    #————————————————————augmentations————————————————————
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='val'))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 32
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/ctrgcn/ctrgcn_gym/j_clip50'
