import numpy as np
import time
import datetime
from pathlib import Path
import cv2
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from siamrpn.SRPN import SiameseRPN
from siamrpn.losses import SmoothL1Loss, Myloss
from siamrpn.utils import x1y1x2y2_to_xywh, xywh_to_x1y1x2y2, clip_anchor

from config import cfg
from dataset_provider import get_dataloader

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a Siam-RPN network')
    parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=0, type=int)
    parser.add_argument('--net', dest='net',
                      default='alexnet', type=str)
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=50, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=1000, type=int)
    # parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
    #                   help='number of iterations to display',
    #                   default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="-",
                      type=str)
    parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=4, type=int)                    
    parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
    # parser.add_argument('--bs', dest='batch_size',
    #                   help='batch_size',
    #                   default=1, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
    # parser.add_argument('--lr_decay_step', dest='lr_decay_step',
    #                   help='step to do learning rate decay, unit is epoch',
    #                   default=5, type=int)
    # parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
    #                   help='learning rate decay ratio',
    #                   default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=0, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--f', dest='finetune',
                      help='finetune or not',
                      default=False, type=bool)
    parser.add_argument('--load_name', dest='load_name',
                      help='path to the loading model',
                      type=str)
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=0, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=0, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

    args = parser.parse_args()
    return args

def imageConvertor(tensor):
    tmp = tensor[0].detach().cpu().numpy().transpose(1,2,0)
    tmp = (tmp * (np.array([0.229, 0.224, 0.225]).reshape(1,1,3))+(np.array([0.485, 0.456, 0.406]).reshape(1,1,3))) * 255
    return tmp.astype(np.uint8)

def getAnchor(a,b,c,anchor_shape):
    return [7+a*15, 7+b*15, anchor_shape[c][0], anchor_shape[c][1]]

def outputConvertor(coutput, routput, anchor_shape):
    coutput = coutput.detach().cpu().numpy()
    routput = routput.detach().cpu().numpy().reshape(5,4,17,17)
    bboxList = []
    maxProb = 0
    for a in range(17):
        for b in range(17):
            for c in range(5):
                anchor = [7+15*a, 7+15*b, anchor_shape[c][0], anchor_shape[c][1]]
                anchor_x1y1x2y2 = xywh_to_x1y1x2y2(anchor)
                anchor_x1y1x2y2 = clip_anchor(anchor_x1y1x2y2)
                anchor = x1y1x2y2_to_xywh(anchor_x1y1x2y2)
                delta = coutput[c,0,a,b]-coutput[c,1,a,b]
                if delta > 10:
                    prob = 0
                elif delta < -10:
                    prob = 1
                else:
                    prob = 1/(1+math.exp(delta))
                maxProb = max(maxProb, prob)
                if prob >= 0.5:
                # if coutput[c,1,a,b]>0.2:
                    bbox = [0,0,0,0]
                    channel0, channel1, channel2, channel3 = routput[c,:,a,b]
                    bbox[0] = channel0*anchor[2] + anchor[0]
                    bbox[1] = channel1*anchor[3] + anchor[1]
                    bbox[2] = math.exp(channel2)*anchor[2]
                    bbox[3] = math.exp(channel3)*anchor[3]
                    bbox = xywh_to_x1y1x2y2(bbox)
                    try:
                        bbox = np.array(bbox, dtype=np.int32)
                    except:
                        print(channel0, channel1, channel2, channel3)
                        print(bbox)
                        continue
                        # raise OverflowError
                    bboxList.append((bbox, prob))


                # if anchor_x1y1x2y2[0]>=0 and anchor_x1y1x2y2[1]>=0 and anchor_x1y1x2y2[2]<=255 and anchor_x1y1x2y2[3]<=255:
                #     try:
                #         delta = coutput[c,0,a,b]-coutput[c,1,a,b]
                #         if delta > 10:
                #             prob = 0
                #         elif delta < -10:
                #             prob = 1
                #         else:
                #             prob = 1/(1+math.exp(delta))
                #         maxProb = max(maxProb, prob)
                #     except:
                #         print(coutput[c,0,a,b], coutput[c,1,a,b], c, a, b)
                #         raise OverflowError
                    # if prob >= 0.5:
                    #     bbox = [0,0,0,0]
                    #     channel0, channel1, channel2, channel3 = routput[c,:,a,b]
                    #     bbox[0] = channel0*anchor[2] + anchor[0]
                    #     bbox[1] = channel1*anchor[3] + anchor[1]
                    #     bbox[2] = math.exp(channel2)*anchor[2]
                    #     bbox[3] = math.exp(channel3)*anchor[3]
                    #     bbox = xywh_to_x1y1x2y2(bbox)
                    #     try:
                    #         bbox = np.array(bbox, dtype=np.int32)
                    #     except:
                    #         print(channel0, channel1, channel2, channel3)
                    #         print(bbox)
                    #         continue
                    #         # raise OverflowError
                    #     bboxList.append(bbox)
    bboxList.sort(key=lambda tup: tup[1], reverse=True)
    return list(map(lambda tup: tup[0], bboxList))[:5], maxProb
    # return bboxList, maxProb

if __name__ == '__main__':
    args = parse_args()

    # print('Called with args:')
    # print(args)

    # print('Using config:')
    # pprint.pprint(cfg)

    assert torch.cuda.is_available(), "GPU is in need"

    # ------------------------------- get dataloaders
    dataloader, totsteps, datasets = get_dataloader(args.num_workers)

    #--------------------------------- output_dir setup
    datasetName = 'flag-{}'.format(cfg.dataset_name)
    output_dir = cfg.PATH.experiment_dir / datasetName
    output_dir.mkdir(exist_ok=True, parents=True)

    #-------------------------------- get model here
    model = SiameseRPN()
    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    #--------------------------------- loading part
    load_name = output_dir / args.load_name
    load_name = str(load_name)
    print("loading checkpoint {}".format(load_name))
    checkpoint = torch.load(load_name)
    # args.session = checkpoint['session']
    # args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr = optimizer.param_groups[0]['lr']
    print("loaded checkpoint {}".format(load_name))
    
    #--------------------------------- main part
    model.eval()
    phase = 'train'

    epoch = 0
    epoch_loss = 0
    epoch_closs = 0
    epoch_rloss = 0

    epoch_size = 0
    imgs = []
    for step, data in enumerate(dataloader[phase]):
        template, detection, clabel, rlabel = data
        target = torch.zeros(clabel.shape).cuda() + 1
        template = Variable(template.cuda())
        detection = Variable(detection.cuda())
        clabel = Variable(clabel.cuda())
        rlabel = Variable(rlabel.cuda())

        # forward
        coutput, routput, ckernal, rkernal, conv1weight, template_features, cinput, detection_features = model(template, detection, debug=True)

        coutput, clabel = coutput.squeeze(), clabel.squeeze()
        coutput = coutput.view(5, 2, 17, 17)              # Batch*k*2*17*17

        # for visualization
        bboxList, maxProb = outputConvertor(coutput, routput, datasets[phase].anchor_shape)
        img = imageConvertor(detection)
        # print("img shape:", img.shape)

        print(len(bboxList), maxProb)
        for idx,bbox in enumerate(reversed(bboxList)):
            if idx < 2:
                img = cv2.rectangle(img.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255), 1)
            else:
                img = cv2.rectangle(img.copy(), (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), idx-1)

        # cv2.imwrite("img.png", img[:,:,::-1])
        imgs.append(img)

        closs = nn.CrossEntropyLoss()(coutput, clabel)
        rloss = SmoothL1Loss(use_gpu = True)(clabel, target, routput, rlabel)
        loss = Myloss()(coutput, clabel, target, routput, rlabel, cfg.lmbda)

        epoch_loss += loss.item()
        epoch_closs += closs.item()
        epoch_rloss += rloss.item()

        epoch_size += 1

        # if step % args.disp_interval == 0:
        #     print("{} step:{} loss:{:.4g} closs:{:.4g} rloss:{:.4g}".format(
        #             phase, step,
        #             epoch_loss/epoch_size, epoch_closs/epoch_size, epoch_rloss/epoch_size,
        #         ))

        if step==63:
            break

    epoch_loss /= epoch_size
    epoch_closs /= epoch_size
    epoch_rloss /= epoch_size

    print('Finish {} loss:{:4g} closs:{:4g} rloss:{:4g}'.format(
            phase, epoch_loss, epoch_closs, epoch_rloss,
        ))

    from torchvision.utils import make_grid
    g = np.stack(imgs)
    g = torch.Tensor(g).byte()
    g = make_grid(g.permute(0,3,1,2))
    g = g.permute(1,2,0).numpy()
    cv2.imwrite("grid.png", g[:,:,::-1]) 

    from IPython import embed
    embed()
    

    