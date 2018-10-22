from easydict import EasyDict as edict
from pathlib import Path

__C = edict()

__C.PATH = edict()
__C.PATH.source_imgs_dir = Path('/home/huangyucheng/MYDATA/DATASETS/PASCAL_VOC/VOCdevkit/VOC2007/JPEGImages')
__C.PATH.root_dir = Path('/unsullied/sharefs/huangyucheng/data/dl-github/Siamese-RPN')
# __C.PATH.train_dir = __C.PATH.root_dir / 'data' / '64scale_train_dataset' / 'train'
__C.PATH.experiment_dir = __C.PATH.root_dir / 'experiment'

# __C.dataset_name = '64scale_train_dataset'
# __C.dataset_name = 'random_insert_10loop'

# __C.anchor_scale = 85
# __C.template_size = 96
# __C.detection_size = 340
# __C.grid_len = 20
__C.anchor_scale = None
__C.template_size = None
__C.detection_size = None
__C.grid_len = None

__C.lmbda = 1

__C.pos_iou_thresh = 0.5
__C.neg_iou_thresh = 0.2

__C.TRAIN = edict()
__C.TRAIN.momentum = 0.9
# __C.TRAIN.batch_size=1

__C.TEST = edict()
__C.TEST.max_validation = 3000

cfg = __C