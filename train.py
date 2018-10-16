import numpy as np
import time
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from siamrpn.SRPN import SiameseRPN
from siamrpn.losses import SmoothL1Loss, Myloss

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
    parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="adam", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=40, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=0, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      action='store_true')
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
    parser.add_argument('--w', dest='weight_decay',
                      help='starting learning rate',
                      default=1e-5, type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # print('Using config:')
    # pprint.pprint(cfg)

    assert torch.cuda.is_available(), "GPU is in need"

    # ------------------------------- get dataloaders
    dataloader, totsteps, datasets = get_dataloader(args.num_workers, args.batch_size)

    #--------------------------------- output_dir setup
    datasetName = 'flag-{}'.format(cfg.dataset_name)
    output_dir = cfg.PATH.experiment_dir / datasetName
    output_dir.mkdir(exist_ok=True, parents=True)

    #-------------------------------- get model here
    model = SiameseRPN()
    model = model.cuda()

    fix_layers = [0,3,6]
    for i in fix_layers:
        layer = model.features[i]
        for param in layer.parameters():
            param.requires_grad = False

    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)
    #--------------------------------- optimizer setup
    lr = args.lr
    if args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, eps=1e-8, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=cfg.TRAIN.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    #--------------------------------- loading part
    if args.resume:
        load_name = output_dir / args.load_name
        load_name = str(load_name)
        print("loading checkpoint {}".format(load_name))
        checkpoint = torch.load(load_name)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("loaded checkpoint {}".format(load_name))
        args.save_dir = args.load_name.split('/')[0]
    elif args.finetune:
        raise NotImplementedError
    else:
        pass

    #--------------------------------- logging part
    if not args.resume:
        args.save_dir = "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")) + args.save_dir + "{}_{}{}_{}{}-{}-{}_{}{}_{}{}_{}{}".format(
                args.net, 
                'opt', args.optimizer,
                'lr', args.lr,args.lr_decay_step,args.lr_decay_gamma,
                'ses',args.session,
                'lmbda',cfg.lmbda,
                'w',args.weight_decay,
            )
        (output_dir / args.save_dir / 'models').mkdir(exist_ok=True, parents=True)
        (output_dir / args.save_dir / 'logs').mkdir(exist_ok=True, parents=True)
    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        train_tb = SummaryWriter(str(output_dir/args.save_dir/'logs'/'train'))
        val_tb = SummaryWriter(str(output_dir/args.save_dir/'logs'/'validation'))
    #--------------------------------- training part
    if args.mGPUs:
        model = torch.nn.DataParallel(model)

    for epoch in range(args.start_epoch, args.max_epochs):
        for phase in ['train','validation']:
            if phase == 'train':
                scheduler.step()
                model.train()
                logger = train_tb
            else:
                model.eval()
                logger = val_tb

            epoch_loss = 0
            epoch_closs = 0
            epoch_rloss = 0

            epoch_size = 0
            start = time.time()
            epoch_start = start
            for step, data in enumerate(dataloader[phase]):
                if phase == 'validation' and step > cfg.TEST.max_validation:
                    break
                template, detection, clabel, rlabel = data
                target = torch.zeros(clabel.shape).cuda() + 1
                template = Variable(template.cuda())
                detection = Variable(detection.cuda())
                clabel = Variable(clabel.cuda())
                rlabel = Variable(rlabel.cuda())
                dltime = time.time()

                optimizer.zero_grad()

                # forward
                coutput, routput = model(template, detection)

                coutput, clabel = coutput.squeeze(), clabel.squeeze()
                coutput = coutput.view(5, 2, 17, 17)              # Batch*k*2*17*17

                closs = nn.CrossEntropyLoss()(coutput, clabel)
                rloss = SmoothL1Loss(use_gpu = True)(clabel, target, routput, rlabel)
                loss = Myloss()(coutput, clabel, target, routput, rlabel, cfg.lmbda)
                # loss, closs, rloss = Myloss()(coutput, clabel, routput, rlabel, cfg.lmbda)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                epoch_closs += closs.item()
                epoch_rloss += rloss.item()

                epoch_size += 1

                if args.use_tfboard:
                    info = {
                        'loss': loss.item(),
                        'rloss': rloss.item(),
                        'closs': closs.item(),
                        'learning_rate': scheduler.get_lr()[0],
                    }
                    if phase=='train':
                        totStep = totsteps[phase]
                    else:
                        totStep = min(totsteps[phase], cfg.TEST.max_validation)
                    for k,v in info.items():
                        logger.add_scalar(k,v,(epoch*totStep + step)*totsteps['train']//totsteps[phase])
                end = time.time()
                if step % args.disp_interval == 0:
                    print("{} e:{}/{} step:{} lr:{:.3g} loss:{:.4g} closs:{:.4g} rloss:{:.4g} dload:{:.3g}".format(
                            phase, epoch,args.max_epochs, step, scheduler.get_lr()[0],
                            epoch_loss/epoch_size, epoch_closs/epoch_size, epoch_rloss/epoch_size,
                            (dltime-start)/(end-start)
                        ))
                start = time.time()

            epoch_loss /= epoch_size
            epoch_closs /= epoch_size
            epoch_rloss /= epoch_size

            print('Finish {} e:{} loss:{:4g} closs:{:4g} rloss:{:4g} time:{:3g}'.format(
                    phase, epoch, epoch_loss, epoch_closs, epoch_rloss,
                    start - epoch_start
                ))

            if phase == 'train':
                save_name = str(output_dir/args.save_dir/'models'/"{}_{}_{}.pth".format(args.session, epoch, step))
                state = {
                    'session': args.session,
                    'epoch': epoch,
                    'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, save_name)
                torch.save(state, str(output_dir/args.save_dir/'models'/'latest'))
                print('save model: {}'.format(save_name))

            print('-'*20)

    if args.use_tfboard:
        train_tb.close()
        val_tb.close()

    from IPython import embed
    embed()

    





