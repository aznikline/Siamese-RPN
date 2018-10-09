from easydict import EasyDict as edict
from pathlib import Path

__C = edict()

__C.PATH = edict()
__C.PATH.source_imgs_dir = Path('/home/huangyucheng/MYDATA/DATASETS/PASCAL_VOC/VOCdevkit/VOC2007/JPEGImages')
__C.PATH.root_dir = Path('/unsullied/sharefs/huangyucheng/data/dl-github/Siamese-RPN')
__C.PATH.train_dir = __C.PATH.root_dir / 'data' / '64scale_train_dataset' / 'train'
__C.PATH.experiment_dir = __C.PATH.root_dir / 'experiment'

__C.template_size = 127
__C.detection_size = 255
__C.lmbda = 1

__C.TRAIN = edict()
__C.TRAIN.momentum = 0.9
__C.TRAIN.batch_size=1

cfg = __C