from easydict import EasyDict as edict
from pathlib import Path

__C = edict()

__C.PATH = edict()
__C.PATH.source_imgs_dir = Path('/home/huangyucheng/MYDATA/DATASETS/PASCAL_VOC/VOCdevkit/VOC2007/JPEGImages')
__C.PATH.root_dir = Path('/unsullied/sharefs/huangyucheng/data/dl-github/Siamese-RPN')
__C.PATH.train_dir = __C.PATH.root_dir / 'data' / '64scale_train_dataset' / 'train'

__C.template_size = 127
__C.detection_size = 255


cfg = __C