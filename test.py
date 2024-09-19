
from dataset.dataset_lits_val import Val_Dataset
from dataset.transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize, Resize_3d, Resize_abs, Crop, Flip_LR
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
import SimpleITK as sitk

import csv
from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict
from models import CDRMamba
import numpy as np
import random
from scipy.spatial.distance import directed_hausdorff
from monai.metrics import HausdorffDistanceMetric

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 在训练开始前设置随机种子
set_seed(2021)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def test(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.Four_metrics_Average(n_labels)
    header = ['DSC', 'JSC', 'PPV', 'RECALL', 'HD95']
    hd95_metric = HausdorffDistanceMetric(percentile=95)
    tensor=torch.randn(1,1,128,128,128)
    DSC_container=[]
    JSC_container = []
    PPV_container = []
    RECALL_container = []
    data = [1.3, 12, 1.5, 122]
    HD95_container = []
    with open('unet_metrics_G.csv', 'w', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        with torch.no_grad():
            for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
                data, target = data.float(), target.long()
                target = common.to_one_hot_3d(target, n_labels)
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss=loss_func(output, target)

                val_loss.update(loss.item(), data.size(0))
                val_dice.update(output, target)
                hd95 = hd95_metric(y_pred=output, y=target).item()
                writer.writerow([val_dice.value1[1], val_dice.value2[1], val_dice.value3[1], val_dice.value4[1]])
                DSC_container.append(val_dice.value1[1])
                JSC_container.append(val_dice.value2[1])
                PPV_container.append(val_dice.value3[1])
                RECALL_container.append(val_dice.value4[1])
                HD95_container.append(hd95)
                print("val_dice = {}".format(val_dice.value1[1]))
                print("val_jsc = {}".format(val_dice.value2[1]))
                print("val_ppv = {}".format(val_dice.value3[1]))
                print("val_recall = {}".format(val_dice.value4[1]))
                print("val_hd95 = {}".format(hd95))
    print("DSC:", sum(DSC_container) / len(DSC_container), "JSC:", sum(JSC_container) / len(JSC_container), "PPV:",
          sum(PPV_container) / len(PPV_container), "RECALL:", sum(RECALL_container) / len(RECALL_container), "HD95:",
          sum(HD95_container) / len(HD95_container))
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_1': val_dice.avg1})
    if n_labels==3: val_log.update({'Val_dice_2': val_dice.avg2})
    return val_log


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./experiments2', args.save)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    val_loader = DataLoader(dataset=Val_Dataset(args),batch_size=1,num_workers=args.n_threads, shuffle=False)

    model = CDRMamba(in_channel=1, out_channel=args.n_labels, training=True).to(device)
    model.apply(weights_init.init_model)
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)
    pth_path = "/home/zxk/checkpoint/"

    ckpt = torch.load(pth_path)
    model.load_state_dict(ckpt['net'],strict=True)
    loss_HD = loss.HausdorffERLoss()
    loss = loss.TverskyLoss()

    best = [0,0]
    trigger = 0
    alpha = 0.4

    val_log = test(model, val_loader, loss, args.n_labels)



    print(val_log['Val_dice_1'])


    torch.cuda.empty_cache()
