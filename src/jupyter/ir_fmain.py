# %%
import time
import argparse
import logging
from tqdm import tqdm
import pandas as pd
from attrdict import AttrDict
from collections import defaultdict
from scipy.stats import gmean
import numpy as np
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from imbalanced_regression.qsm.resnetf import resnetf50
from imbalanced_regression.qsm import loss
from imbalanced_regression.qsm.datasets import QSM_features
from imbalanced_regression.utils import *
import util
from util import pyvis, model_scale
import sklearn.model_selection as skm
import os
import nibabel as nib
from IPython.display import HTML
from datetime import datetime
os.environ["KMP_WARNINGS"] = "FALSE"
HTML('''
<style>
.jupyter-matplotlib {
    background-color: #000;
}

.widget-label, .jupyter-matplotlib-header{
    color: #fff;
}

.jupyter-button {
    background-color: #333;
    color: #fff;
}
</style>
''')
#%matplotlib widget

# %%
# Get case IDs
case_list = open('/home/ali/RadDBS-QSM/data/docs/cases_90','r')
lines = case_list.read()
lists = np.loadtxt(case_list.name,comments="#", delimiter=",",unpack=False,dtype=str)
case_id = []
for lines in lists:     
    case_id.append(lines[-9:-7])

# Load scores
file_dir = '/home/ali/RadDBS-QSM/data/docs/QSM anonymus- 6.22.2023-1528.csv'
motor_df = util.filter_scores(file_dir,'pre-dbs updrs','stim','CORNELL ID')
# Find cases with all required scores
subs,pre_imp,post_imp,pre_updrs_off = util.get_full_cases(motor_df,
                                                          'CORNELL ID',
                                                          'OFF (pre-dbs updrs)',
                                                          'ON (pre-dbs updrs)',
                                                          'OFF meds ON stim 6mo')
# Find overlap between scored subjects and nii
ids = np.asarray(case_id).astype(int)
# ids = ids[ids != 54]
cases_idx = np.in1d(subs,ids)
ccases = subs[cases_idx]
per_change = np.round(post_imp[cases_idx],1)

nii_paths = []
per_change_match = []
updrs_match = []
qsm_dir = '/home/ali/RadDBS-QSM/data/npy/X/'
qsm_niis = sorted(os.listdir(qsm_dir))
for k in np.arange(len(ccases)):
    for file in qsm_niis:
        if int(ccases[k]) == int(file[4:6]):
            nii_paths.append(qsm_dir+file)
            per_change_match.append(per_change[k])
            updrs_match.append(pre_updrs_off[k])

train_dir, test_dir, y_train, y_test, pre_train, pre_test = skm.train_test_split(nii_paths, per_change_match, updrs_match, test_size=0.1, random_state=2)
train_dir, val_dir, y_train, y_val, pre_train, pre_val = skm.train_test_split(train_dir, y_train, pre_train, test_size=0.2, random_state=2)

# %%
args = AttrDict()
args.gpu = 0
args.optimizer = 'sgd'
args.lr = 1e-3
args.epoch = 100
args.momentum = 0.9
args.weight_decay = 1e-4
args.schedule = [60,80]
args.print_freq = 10
args.resume = ''
args.pretrained = False
args.evaluate = False
args.loss = 'l1'
args.dataset = 'qsm'
args.model = 'resnet50'
args.store_root = '/home/ali/RadDBS-QSM/data/checkpoint'
args.data_dir = '/home/ali/RadDBS-QSM/data/qsm/'
args.fds = False
args.fds_kernel = 'gaussian'
args.fds_ks = 3
args.fds_sigma = 1
args.fds_mmt = 0.9
args.start_update = 0
args.start_smooth = 1
args.bucket_num = 20
args.bucket_start = 3
args.start_epoch = 0
args.best_loss = 1e5
args.reweight = 'sqrt_inv'
args.retrain_fc = False
args.lds = True
args.lds_kernel = 'gaussian'
args.lds_ks = 3
args.lds_sigma = 1
args.batch_size = 11
args.workers = 24
args.input_type = 'features'

args.store_name = 'L_'+str(args.loss)+'_FDS_'+str(args.fds)+'_LDS_'+str(args.lds)+'_BS_'+str(args.batch_size)+'_IT_'+str(args.input_type)+'_'+str(datetime.now())
# %%
def main():
    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    # Data
    print('=====> Preparing data...')

    train_dataset = QSM_features(data_dir=train_dir, targets=y_train, pre_metric=pre_train, scaler_ss=None, split='train',
                          reweight=args.reweight, lds=args.lds, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    val_dataset = QSM_features(data_dir=val_dir, targets=y_val, pre_metric=pre_val, scaler_ss=train_dataset.__getscaler__(), split='val')
    test_dataset = QSM_features(data_dir=test_dir, targets=y_test, pre_metric=pre_test, scaler_ss=train_dataset.__getscaler__(), split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True, drop_last=False)
    print(f"Training data size: {len(train_dataset)}")
    print(f"Validation data size: {len(val_dataset)}")
    print(f"Test data size: {len(test_dataset)}")
    print(y_test)

    # Model
    print('=====> Building model...')
    model = resnetf50(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
                     start_update=args.start_update, start_smooth=args.start_smooth,
                     kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)
    model = torch.nn.DataParallel(model).cuda()
    print(model)
    # evaluate only
    if args.evaluate:
        assert args.resume, 'Specify a trained model using [args.resume]'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"===> Checkpoint '{args.resume}' loaded (epoch [{checkpoint['epoch']}]), testing...")
        validate(test_loader, model, train_labels=y_test, prefix='Test')
        return

    if args.retrain_fc:
        assert args.reweight != 'none' and args.pretrained
        print('===> Retrain last regression layer only!')
        for name, param in model.named_parameters():
            if 'fc' not in name and 'linear' not in name:
                param.requires_grad = False

    # Loss and optimizer
    if not args.retrain_fc:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        # optimize only the last linear layer
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        names = list(filter(lambda k: k is not None, [k if v.requires_grad else None for k, v in model.module.named_parameters()]))
        assert 1 <= len(parameters) <= 2  # fc.weight, fc.bias
        print(f'===> Only optimize parameters: {names}')
        optimizer = torch.optim.Adam(parameters, lr=args.lr) if args.optimizer == 'adam' else \
            torch.optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'linear' not in k and 'fc' not in k:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f'===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]')
        print(f'===> Pre-trained model loaded: {args.pretrained}')

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"===> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume) if args.gpu is None else \
                torch.load(args.resume, map_location=torch.device(f'cuda:{str(args.gpu)}'))
            args.start_epoch = checkpoint['epoch']
            args.best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"===> Loaded checkpoint '{args.resume}' (Epoch [{checkpoint['epoch']}])")
        else:
            print(f"===> No checkpoint found at '{args.resume}'")

    globals()[f"weighted_{args.loss}_loss"] = loss.weighted_l1_loss
    cudnn.benchmark = True
    os.mkdir(args.store_root+'/'+args.store_name)

    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(train_loader, model, optimizer, epoch)
        val_loss_mse, val_loss_l1, val_loss_gmean = validate(val_loader, model, train_labels=y_train)
        loss_metric = val_loss_mse if args.loss == 'mse' else val_loss_l1
        is_best = loss_metric < args.best_loss
        args.best_loss = min(loss_metric, args.best_loss)
        print(f"Best {'L1' if 'l1' in args.loss else 'MSE'} Loss: {args.best_loss:.3f}")
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'model': args.model,
            'best_loss': args.best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best)
        print(f"Epoch #{epoch}: Train loss [{train_loss:.4f}]; "
              f"Val loss: MSE [{val_loss_mse:.4f}], L1 [{val_loss_l1:.4f}], G-Mean [{val_loss_gmean:.4f}]")

    # test with best checkpoint
    print("=" * 120)
    print("Test best model on testset...")
    checkpoint = torch.load(f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['best_loss']:.4f}")
    test_loss_mse, test_loss_l1, test_loss_gmean = validate(test_loader, model, train_labels=y_train, prefix='Test')
    print(f"Test loss: MSE [{test_loss_mse:.4f}], L1 [{test_loss_l1:.4f}], G-Mean [{test_loss_gmean:.4f}]\nDone")

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.4f')
    losses = AverageMeter(f'Loss ({args.loss.upper()})', ':.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch)
    )

    model.train()
    end = time.time()
    for idx, (inputs, targets, weights) in enumerate(train_loader):
        data_time.update(time.time() - end)
        inputs, targets, weights = \
            inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True), weights.cuda(non_blocking=True)
        if args.fds:
            outputs, _ = model(inputs, targets, epoch)
            print('Predicted ',str(outputs),'for true improvement ',str(targets))
        else:
            outputs = model(inputs, targets, epoch)
            #print('Predicted ',str(outputs),'for true improvement ',str(targets))
        loss = globals()[f"weighted_{args.loss}_loss"](outputs, torch.unsqueeze(targets,dim=1), weights)
        assert not (np.isnan(loss.item()) or loss.item() > 1e6), f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if idx % args.print_freq == 0:
            progress.display(idx)

    if args.fds and epoch >= args.start_update:
        print(f"Create Epoch [{epoch}] features of all training data...")
        encodings, labels = [], []
        with torch.no_grad():
            for (inputs, targets, _) in tqdm(train_loader):
                inputs = inputs.cuda(non_blocking=True)
                outputs, feature = model(inputs, targets, epoch)
                encodings.extend(feature.data.squeeze().cpu().numpy())
                labels.extend(targets.data.squeeze().cpu().numpy())

        encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
        model.module.FDS.update_last_epoch_stats(epoch)
        model.module.FDS.update_running_stats(encodings, labels, epoch)

    return losses.avg


def validate(val_loader, model, train_labels=None, prefix='Val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses_mse, losses_l1],
        prefix=f'{prefix}: '
    )

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')

    model.eval()
    losses_all = []
    preds, labels = [], []
    with torch.no_grad():
        end = time.time()
        for idx, (inputs, targets, _) in enumerate(val_loader):
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            outputs = model(torch.squeeze(inputs,dim=1))

            preds.extend(outputs.data.cpu().numpy())
            labels.extend(targets.data.cpu().numpy())
            loss_mse = criterion_mse(outputs, torch.unsqueeze(targets,dim=1))
            loss_l1 = criterion_l1(outputs, torch.unsqueeze(targets,dim=1))
            loss_all = criterion_gmean(outputs, torch.unsqueeze(targets,dim=1))
            losses_all.extend(loss_all.cpu().numpy())

            losses_mse.update(loss_mse.item(), inputs.size(0))
            losses_l1.update(loss_l1.item(), inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if idx % args.print_freq == 0:
                progress.display(idx)
        # print('Validate labels: ',str(labels))
        # print('Calculating shot metrics for validation predictions of size ',str(len(preds)),' with validations labels of size ',str(len(labels)), ' and training labels of size ', str(len(train_labels)))
        # print('Train labels: ',str(train_labels))
        shot_dict = shot_metrics(np.hstack(preds), np.hstack(labels), train_labels)
        loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
        print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
        print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
              f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
        # print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
        #       f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
        print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
              f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")
        print('Predicted ',str(list(np.asarray(preds).ravel())),' for true improvements ',str(list(labels)))
        #print('Predicted ',str(list(outputs)),' for true improvements ',str(list(targets)))
    return losses_mse.avg, losses_l1.avg, loss_gmean


def shot_metrics(preds, labels, train_labels, many_shot_thr=1):#, low_shot_thr=1):
   
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    #print('Unique validate labels: ',str(np.unique(labels)))
    #print(train_labels)
    for l in np.unique(labels):
        # print('Looking for label ',str(l),' in train labels: ',str(train_labels))
        train_class_count.append(len(np.asarray(train_labels)[train_labels == l]))
        #print(train_labels == l)
        # print('Looking for label ',str(l),' in test labels: ',str(labels))
        test_class_count.append(len(labels[labels == l]))
        # print('Train class count is ',str(train_class_count))
        # print('Test class count is ',str(test_class_count))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        else:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        # elif train_class_count[i] < low_shot_thr:
        #     low_shot_mse.append(mse_per_class[i])
        #     low_shot_l1.append(l1_per_class[i])
        #     low_shot_gmean += list(l1_all_per_class[i])
        #     low_shot_cnt.append(test_class_count[i])
        # else:
        #     median_shot_mse.append(mse_per_class[i])
        #     median_shot_l1.append(l1_per_class[i])
        #     median_shot_gmean += list(l1_all_per_class[i])
        #     median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    # shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    # shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    # shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict


if __name__ == '__main__':
    main()




