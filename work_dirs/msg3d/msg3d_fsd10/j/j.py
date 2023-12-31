model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='MSG3D', graph_cfg=dict(layout='openpose_25', mode='binary_adj')),
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=384))
dataset_type = 'PoseDataset'
ann_file = 'data/fsd10/fsd10.pkl'
train_pipeline = [
    dict(type='GenSkeFeat', dataset='openpose_25', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset='openpose_25', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset='openpose_25', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
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
        times=200,
        dataset=dict(
            type='PoseDataset',
            ann_file='data/fsd10/fsd10.pkl',
            pipeline=[
                dict(type='GenSkeFeat', dataset='openpose_25', feats=['j']),
                dict(type='UniformSample', clip_len=100),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=2),
                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['keypoint'])
            ],
            split='train')),
    val=dict(
        type='PoseDataset',
        ann_file='data/fsd10/fsd10.pkl',
        pipeline=[
            dict(type='GenSkeFeat', dataset='openpose_25', feats=['j']),
            dict(type='UniformSample', clip_len=100, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='val'),
    test=dict(
        type='PoseDataset',
        ann_file='data/fsd10/fsd10.pkl',
        pipeline=[
            dict(type='GenSkeFeat', dataset='openpose_25', feats=['j']),
            dict(type='UniformSample', clip_len=100, num_clips=10),
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
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/msg3d/msg3d_fsd10/j'
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)
resume_from = './work_dirs/msg3d/msg3d_fsd10/j/latest.pth'
