model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN', graph_cfg=dict(layout='openpose_25', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=10, in_channels=256))
dataset_type = 'PoseDataset'
ann_file = 'data/fsd10/fsd10_kbc.pkl'
train_pipeline = [
    dict(type='GenSkeFeat', dataset='openpose_25', feats=['b']),
    dict(
        type='UniformSample',
        clip_len=100,
        keyframe_temporal_segment=dict(
            keyframes=25, temporal_segment=False, resort=True)),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='GenSkeFeat', dataset='openpose_25', feats=['b']),
    dict(
        type='UniformSample',
        clip_len=100,
        num_clips=1,
        keyframe_temporal_segment=dict(
            keyframes=25, temporal_segment=False, resort=True)),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='GenSkeFeat', dataset='openpose_25', feats=['b']),
    dict(
        type='UniformSample',
        clip_len=100,
        num_clips=10,
        keyframe_temporal_segment=dict(
            keyframes=25, temporal_segment=False, resort=True)),
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
        times=10,
        dataset=dict(
            type='PoseDataset',
            ann_file='data/fsd10/fsd10_kbc.pkl',
            pipeline=[
                dict(type='GenSkeFeat', dataset='openpose_25', feats=['b']),
                dict(
                    type='UniformSample',
                    clip_len=100,
                    keyframe_temporal_segment=dict(
                        keyframes=25, temporal_segment=False, resort=True)),
                dict(type='PoseDecode'),
                dict(type='FormatGCNInput', num_person=2),
                dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
                dict(type='ToTensor', keys=['keypoint'])
            ],
            split='train')),
    val=dict(
        type='PoseDataset',
        ann_file='data/fsd10/fsd10_kbc.pkl',
        pipeline=[
            dict(type='GenSkeFeat', dataset='openpose_25', feats=['b']),
            dict(
                type='UniformSample',
                clip_len=100,
                num_clips=1,
                keyframe_temporal_segment=dict(
                    keyframes=25, temporal_segment=False, resort=True)),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=2),
            dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint'])
        ],
        split='val'),
    test=dict(
        type='PoseDataset',
        ann_file='data/fsd10/fsd10_kbc.pkl',
        pipeline=[
            dict(type='GenSkeFeat', dataset='openpose_25', feats=['b']),
            dict(
                type='UniformSample',
                clip_len=100,
                num_clips=10,
                keyframe_temporal_segment=dict(
                    keyframes=25, temporal_segment=False, resort=True)),
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
total_epochs = 64
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/ctrgcn/ctrgcn_fsd10_keyframesample/b3'
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)
