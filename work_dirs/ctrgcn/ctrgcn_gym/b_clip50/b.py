model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN', graph_cfg=dict(layout='coco', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=99, in_channels=256))
dataset_type = 'PoseDataset'
ann_file = 'data/gym/gym_hrnet.pkl'
train_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
    dict(type='UniformSample', clip_len=50),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
    dict(type='UniformSample', clip_len=50, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset='coco', feats=['b']),
    dict(type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
    dict(type='UniformSample', clip_len=50, num_clips=10),
    dict(type='PoseDecode'),
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
        dataset=dict(
            type='PoseDataset',
            ann_file='data/gym/gym_hrnet.pkl',
            pipeline=[
                dict(type='GenSkeFeat', dataset='coco', feats=['b']),
                dict(
                    type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
                dict(type='UniformSample', clip_len=50),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=2),
                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['keypoint'])
            ],
            split='train')),
    val=dict(
        type='PoseDataset',
        ann_file='data/gym/gym_hrnet.pkl',
        pipeline=[
            dict(type='GenSkeFeat', dataset='coco', feats=['b']),
            dict(type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
            dict(type='UniformSample', clip_len=50, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='val'),
    test=dict(
        type='PoseDataset',
        ann_file='data/gym/gym_hrnet.pkl',
        pipeline=[
            dict(type='GenSkeFeat', dataset='coco', feats=['b']),
            dict(type='PreNormalize2D', mode='fix', img_shape=(1080, 1920)),
            dict(type='UniformSample', clip_len=50, num_clips=10),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='val'))
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 32
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/ctrgcn/ctrgcn_gym/b_clip50'
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)
resume_from = './work_dirs/ctrgcn/ctrgcn_gym/b_clip50/latest.pth'
