from flag.builder import FlagBuilder
from config import cfg
from siamrpn.utils import x1y1x2y2_to_xywh, xywh_to_x1y1x2y2, clip_anchor

import json
# import cv2
from PIL import Image
import math
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MyDataset(Dataset):

    def __init__(self, dataset_name, anchor_scale = 64, k = 5, phase='train'):
        self.anchor_shape = self._get_anchor_shape(anchor_scale)
        self.k = k
        self.grid_len = cfg.grid_len
        self.phase = phase
        self.path = cfg.PATH.root_dir / 'data' / dataset_name / phase
        self.infoList = json.loads(open(str(self.path / 'infoList.json')).read())
        self.load_gallery()
        self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        
    """根据anchor_scale获得5个anchor的宽度和高度
    """
    def _get_anchor_shape(self, a):
        s = a**2
        r = [[3*math.sqrt(s/3.),math.sqrt(s/3.)], [2*math.sqrt(s/2.),math.sqrt(s/2.)], 
                 [a,a], [math.sqrt(s/2.),2*math.sqrt(s/2.)], [math.sqrt(s/3.),3*math.sqrt(s/3.)]]
        return [list(map(round, i)) for i in r]

    def __len__(self):
        return len(self.infoList)
    
    def load_image(self, img_path):
        # img = cv2.imread(img_path)
        # img = np.array(Image.open(img_path))
        img = Image.open(str(img_path))
        if img.mode != "RGB":
            img = img.convert('RGB')
        return img

    def load_gallery(self):
        if self.phase == 'validation':
            gallery_dir = self.path.parent / 'train' / 'gallery'
        else:
            gallery_dir = self.path/'gallery'
        gallery = {}
        for path in gallery_dir.glob('*'):
            ind = int(path.name.split('.')[0])
            img = self.load_image(path)
            img = img.resize((cfg.template_size, cfg.template_size), Image.BILINEAR)
            gallery[ind] = img
        self.gallery = gallery

    """读取数据集时，将会调用下面这个方法来获取数据
    """
    def __getitem__(self, index):
        info = self.infoList[index]

        img_path = info['path']
        img = self.load_image(img_path)
        original_size = img.size
        img = img.resize((cfg.detection_size, cfg.detection_size), Image.BILINEAR)
        bbox = info['bbox'][:]
        self.resize_bbox(bbox, original_size)
        gtbox = x1y1x2y2_to_xywh(bbox)

        label = info['label']
        template = self.gallery[label]

        # if self.phase == 'train':
        #     clabel, rlabel = self._gtbox_to_label(gtbox)
        # else:
        #     clabel, rlabel = self._get_test_label(gtbox)
        clabel, rlabel = self._get_label(gtbox)
        return self.transforms(template), self.transforms(img), clabel, rlabel

    def resize_bbox(self, bbox, original_size):
        bbox[0] *= cfg.detection_size/original_size[0]
        bbox[2] *= cfg.detection_size/original_size[0]
        bbox[1] *= cfg.detection_size/original_size[1]
        bbox[3] *= cfg.detection_size/original_size[1]
    
    '''数据转换，包括裁剪、变形、转换为tensor、归一化
    '''
    def _transform(self, img, gtbox, area, size):
        img, pcc = point_center_crop(img, gtbox, area)
        img, ratio = resize(img, size)
        img = F.to_tensor(img)
#        img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, pcc, ratio
        
    """根据ground truth box构造class label和reg label
    """
    def _get_label(self, gtbox):
        clabel = np.zeros([5, 17, 17]) - 100
        rlabel = np.zeros([20, 17, 17], dtype = np.float32)
        if self.phase in ['train','validation']:
            pos, neg = self._get_64_anchors_new(gtbox)
            assert len(pos)+len(neg)==64
            for a,b,c,anchor_x1y1x2y2 in pos:
                anchor = x1y1x2y2_to_xywh(anchor_x1y1x2y2)
                clabel[c,a,b] = 1
                channel0 = (gtbox[0] - anchor[0])/anchor[2]
                channel1 = (gtbox[1] - anchor[1])/anchor[3]
                channel2 = math.log(gtbox[2]/anchor[2])
                channel3 = math.log(gtbox[3]/anchor[3])
                rlabel[c*4:c*4+4,a,b] = [channel0, channel1, channel2, channel3]
            for a,b,c,anchor in neg:
                clabel[c,a,b] = 0
        else :
            for a in range(17):
                for b in range(17):
                    for c in range(5):
                        anchor = [self.grid_len//2+self.grid_len*a, self.grid_len//2+self.grid_len*b, self.anchor_shape[c][0], self.anchor_shape[c][1]]
                        anchor = xywh_to_x1y1x2y2(anchor)
                        anchor = clip_anchor(anchor)
                        iou = self._IOU(anchor, gtbox)
                        anchor = x1y1x2y2_to_xywh(anchor)
                        if iou >= cfg.pos_iou_thresh:
                            clabel[c,a,b] = 1
                            channel0 = (gtbox[0] - anchor[0])/anchor[2]
                            channel1 = (gtbox[1] - anchor[1])/anchor[3]
                            channel2 = math.log(gtbox[2]/anchor[2])
                            channel3 = math.log(gtbox[3]/anchor[3])
                            rlabel[c*4:c*4+4,a,b] = [channel0, channel1, channel2, channel3]
                        elif iou <= cfg.neg_iou_thresh:
                            clabel[c,a,b] = 0
        return torch.Tensor(clabel).long(), torch.Tensor(rlabel).float()

    # def _gtbox_to_label(self, gtbox):
    #     clabel = np.zeros([5, 17, 17]) - 100
    #     rlabel = np.zeros([20, 17, 17], dtype = np.float32)
    #     pos, neg = self._get_64_anchors(gtbox)
    #     for i in range(len(pos)):
    #         clabel[pos[i, 2], pos[i, 0], pos[i, 1]] = 1
    #     for i in range(len(neg)):
    #         clabel[neg[i, 2], neg[i, 0], neg[i, 1]] = 0
    #     pos_coord = self._anchor_coord(pos)
    #     channel0 = (gtbox[0] - pos_coord[:, 0]) / pos_coord[:, 2]
    #     channel1 = (gtbox[1] - pos_coord[:, 1]) / pos_coord[:, 3]
    #     channel2 = np.array([math.log(i) for i in (gtbox[2] / pos_coord[:, 2]).tolist()])
    #     channel3 = np.array([math.log(i) for i in (gtbox[3] / pos_coord[:, 3]).tolist()])
    #     for i in range(len(pos)):
    #         rlabel[pos[i][2]*4, pos[i][0], pos[i][1]] = channel0[i]
    #         rlabel[pos[i][2]*4 + 1, pos[i][0], pos[i][1]] = channel1[i]
    #         rlabel[pos[i][2]*4 + 2, pos[i][0], pos[i][1]] = channel2[i]
    #         rlabel[pos[i][2]*4 + 3, pos[i][0], pos[i][1]] = channel3[i]
    #     return torch.Tensor(clabel).long(), torch.Tensor(rlabel).float()
    
    # """根据anchor在label中的位置来获取anchor在detection frame中的坐标
    # """
    # def _anchor_coord(self, pos):
    #     result = np.ndarray([0, 4])
    #     for i in pos:
    #         tmp = [self.grid_len//2+self.grid_len*i[0], self.grid_len//2+self.grid_len*i[1], self.anchor_shape[i[2]][0], self.anchor_shape[i[2]][1]]
    #         result = np.concatenate([result, np.array(tmp).reshape([1,4])], axis = 0)
    #     return result

    # def _get_test_label(self, gtbox):
    #     clabel = np.zeros([5,17,17]) - 100
    #     rlabel = np.zeros([20, 17, 17], dtype = np.float32)
    #     dct = {}
    #     for a in range(17):
    #         for b in range(17):
    #             for c in range(5):
    #                 anchor = [self.grid_len//2+self.grid_len*a, self.grid_len//2+self.grid_len*b, self.anchor_shape[c][0], self.anchor_shape[c][1]]
    #                 channel0 = (gtbox[0] - anchor[0])/anchor[2]
    #                 channel1 = (gtbox[1] - anchor[1])/anchor[3]
    #                 channel2 = math.log(gtbox[2]/anchor[2])
    #                 channel3 = math.log(gtbox[3]/anchor[3])
    #                 rlabel[c*4:c*4+4,a,b] = [channel0, channel1, channel2, channel3]
    #                 anchor = xywh_to_x1y1x2y2(anchor)
    #                 if anchor[0]>=0 and anchor[1]>=0 and anchor[2]<=255 and anchor[3]<=255:
    #                     iou = self._IOU(anchor, gtbox)
    #                     if iou >= cfg.pos_iou_thresh:
    #                         clabel[c,a,b] = 1
    #     return torch.Tensor(clabel).long(), torch.Tensor(rlabel).float()

    def _get_64_anchors_new(self, gtbox):
        pos = {}
        neg = {}
        zero = {}
        for a in range(17):
            for b in range(17):
                for c in range(5):
                    anchor = [self.grid_len//2+self.grid_len*a, self.grid_len//2+self.grid_len*b, self.anchor_shape[c][0], self.anchor_shape[c][1]]
                    anchor = xywh_to_x1y1x2y2(anchor)
                    anchor = clip_anchor(anchor)
                    iou = self._IOU(anchor, gtbox)
                    # if iou>0.3:
                        # print(x1y1x2y2_to_xywh(anchor), gtbox, iou)

                    if iou >= cfg.pos_iou_thresh:
                        pos[(a,b,c)] = (iou, anchor)
                    elif iou <= cfg.neg_iou_thresh and iou > 0:
                        neg[(a,b,c)] = (iou, anchor)
                    else :
                        zero[(a,b,c)] = (iou, anchor)
        # print(len(pos), len(neg), len(zero))
        pos = sorted(pos.items(), key=lambda tup: tup[1][0], reverse=True)
        pos = [(*tup[0], tup[1][1]) for tup in pos[:16]]
        num = math.ceil((64-len(pos))/2)
        neg = sorted(neg.items(), key=lambda tup: tup[1][0], reverse=True)
        neg = [(*tup[0], tup[1][1]) for tup in neg[:num]]
        zero = list(zero.items())
        np.random.shuffle(zero)
        neg += [(*tup[0], tup[1][1]) for tup in zero[:64-len(pos)-len(neg)]]
        return pos, neg


    # def _get_64_anchors(self, gtbox):
    #     pos = {}
    #     neg = {}
    #     for a in range(17):
    #         for b in range(17):
    #             for c in range(5):
    #                 anchor = [self.grid_len//2+self.grid_len*a, self.grid_len//2+self.grid_len*b, self.anchor_shape[c][0], self.anchor_shape[c][1]]
    #                 anchor = xywh_to_x1y1x2y2(anchor)
    #                 if anchor[0]>=0 and anchor[1]>=0 and anchor[2]<=255 and anchor[3]<=255:
    #                     iou = self._IOU(anchor, gtbox)
    #                     if iou >= cfg.pos_iou_thresh+0.1:
    #                         pos['%d,%d,%d' % (a,b,c)] = iou
    #                     elif iou >= cfg.neg_iou_thresh and iou < cfg.pos_iou_thresh-0.1:
    #                         neg['%d,%d,%d' % (a,b,c)] = iou
    #     pos = sorted(pos.items(),key = lambda x:x[1],reverse = True)
    #     pos = [list(map(int, i[0].split(','))) for i in pos[:16]]
    #     neg = sorted(neg.items(),key = lambda x:x[1],reverse = True)
    #     neg = [list(map(int, i[0].split(','))) for i in neg[:(64-len(pos))]]
    #     return np.array(pos), np.array(neg)

#    def _f(self, x):
#        if x <= 0:      return 0
#        elif x >= 254:  return 254
#        else:           return x

    def _IOU(self, a, b):
#        a = xywh_to_x1y1x2y2(a)
        b = xywh_to_x1y1x2y2(b)
        sa = (a[2] - a[0]) * (a[3] - a[1]) 
        sb = (b[2] - b[0]) * (b[3] - b[1])
        w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        area = w * h 
        return area / (sa + sb - area)

def get_dataloader(dataset_name,num_workers=0, batch_size=1):
    transformed_dataset_train = MyDataset(dataset_name=dataset_name,anchor_scale=cfg.anchor_scale)
    transformed_dataset_validation = MyDataset(phase='validation',dataset_name=dataset_name, anchor_scale=cfg.anchor_scale)
    transformed_dataset_test = MyDataset(phase='test',dataset_name=dataset_name, anchor_scale=cfg.anchor_scale)
    train_dataloader = DataLoader(transformed_dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_dataloader = DataLoader(transformed_dataset_validation, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(transformed_dataset_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader = {'train':train_dataloader, 'validation':validation_dataloader, 'test': test_dataloader}
    totsteps = {
        'train': len(transformed_dataset_train)//batch_size,
        'validation': len(transformed_dataset_validation)//batch_size,
        'test': len(transformed_dataset_test)//batch_size,
    }
    datasets = {
        'train': transformed_dataset_train,
        'validation': transformed_dataset_validation,
        'test': transformed_dataset_test,
    }
    return dataloader, totsteps, datasets

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', default=False, action="store_true")
    parser.add_argument('--testds', default=False, action="store_true")
    args = parser.parse_args()

    if args.build:
        builder = FlagBuilder(cfg.PATH.root_dir)
        iter_img_paths = cfg.PATH.source_imgs_dir.glob('*.jpg')
        # builder.build_train_dataset("64scale_train_dataset", iter_img_paths, num_train_classes=100, num_test_classes=100, 
        #                             scaleRange=None)
        # builder.build_test_dataset("64scale_train_dataset", iter_img_paths, scaleRange=None)
        # builder.build_val_dataset("64scale_train_dataset", iter_img_paths, scaleRange=None)
        builder.build("3loop_for_resnet", iter_img_paths, exist_ok=False, num_train_classes=100, num_test_classes=100, 
                scaleRange=None, iter_loop=3)

    if args.testds:
        cfg.anchor_scale = 85
        cfg.template_size = 96
        cfg.detection_size = 340
        cfg.grid_len = 20
        ds = MyDataset('3loop_for_resnet')
        from IPython import embed
        embed()

