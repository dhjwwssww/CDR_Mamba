# from dataset.dataset_lits_val import Val_Dataset
# from dataset.dataset_lits_train import Train_Dataset

from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset
from dataset.transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, Resize_3d, Resize_abs, Crop, Flip_LR
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
import SimpleITK as sitk

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
from models import CDRMamba
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 在训练开始前设置随机种子
set_seed(2021)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_1': val_dice.avg[1]})
    if n_labels==3: val_log.update({'Val_dice_2': val_dice.avg[2]})
    return val_log

def train(model, train_loader, optimizer, loss_func, n_labels, alpha,belta,theta):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target,n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss0 = loss_func(output[0], target)
        loss1 = loss_func(output[1], target)
        loss2 = loss_func(output[2], target)
        loss3 = loss_func(output[3], target)

        loss = loss3  +  alpha * (loss0 + belta*(loss1 + theta*loss2))
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss3.item(),data.size(0))
        train_dice.update(output[3], target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_1': train_dice.avg[1]})
    if n_labels==3: val_log.update({'Train_dice_2': train_dice.avg[2]})
    return val_log


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('/home/zxk/checkpoint/', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args),batch_size=args.batch_size,num_workers=args.n_threads, shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(args),batch_size=1,num_workers=args.n_threads, shuffle=False)

    model = CDRMamba(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)

    pth_path = r'/home/zxk/checkpoint/'
    ckpt = torch.load('{}'.format(pth_path))
    model.load_state_dict(ckpt['net'], strict=True)
    loss = loss.TverskyLoss()

    log = logger.Train_Logger(save_path,"train_log")

    best = [0,0] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    a=0.4
    b=0.8
    c=0.2
    alpha = 0.4 # 深监督衰减系数初始值
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, a,b,c)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        latest_model_file_name = 'latest_model_{:.4f}.pth'.format(val_log['Val_dice_1'])
        torch.save(state, os.path.join(save_path, latest_model_file_name))
        trigger += 1
        if val_log['Val_dice_1'] > best[1]:
            print('Saving best model')
            model_file_name = 'best_model_{:.4f}.pth'.format(val_log['Val_dice_1'])
            torch.save(state, os.path.join(save_path, model_file_name))
            best[0] = epoch
            best[1] = val_log['Val_dice_1']
            trigger = 0

        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))


        # 深监督系数衰减
        if epoch % 50 == 0: alpha *= 0.9

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()    