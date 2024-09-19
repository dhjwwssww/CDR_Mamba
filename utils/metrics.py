
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)


class JSC_Average(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            jsc = inter / (union-inter)
            dices.append(jsc.item())
        return np.asarray(dices)

class PPV_Average(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            ppv = inter / torch.sum(logits[:, class_index, :, :, :])
            dices.append(ppv.item())
        return np.asarray(dices)

class RECALL_Average(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])

            recall = inter / (inter + torch.sum(logits[:, 0, :, :, :] * targets[:, 1, :, :, :]))
            dices.append(recall.item())
        return np.asarray(dices)

class Four_metrics_Average(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value1 = np.asarray([0]*self.class_num, dtype='float64')
        self.value2 = np.asarray([0] * self.class_num, dtype='float64')
        self.value3 = np.asarray([0] * self.class_num, dtype='float64')
        self.value4 = np.asarray([0] * self.class_num, dtype='float64')
        self.avg1 = np.asarray([0]*self.class_num, dtype='float64')
        self.avg2 = np.asarray([0] * self.class_num, dtype='float64')
        self.avg3 = np.asarray([0] * self.class_num, dtype='float64')
        self.avg4 = np.asarray([0] * self.class_num, dtype='float64')
        self.sum1 = np.asarray([0]*self.class_num, dtype='float64')
        self.sum2 = np.asarray([0] * self.class_num, dtype='float64')
        self.sum3 = np.asarray([0] * self.class_num, dtype='float64')
        self.sum4 = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value1, self.value2, self.value3, self.value4 = Four_metrics_Average.get_dices(
            logits, targets)
        self.sum1 += self.value1
        self.sum2 += self.value2
        self.sum3 += self.value3
        self.sum4 += self.value4
        self.count += 1
        self.avg1 = np.around(self.sum1 / self.count, 4)
        self.avg2 = np.around(self.sum2 / self.count, 4)
        self.avg3 = np.around(self.sum3 / self.count, 4)
        self.avg4 = np.around(self.sum4 / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        jscs = []
        ppvs = []
        recalls = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            jsc = inter / (union - inter)
            ppv = inter / torch.sum(logits[:, class_index, :, :, :])
            recall = inter / (inter + torch.sum(logits[:, 0, :, :, :] * targets[:, 1, :, :, :]))
            dices.append(dice.item())
            jscs.append(jsc.item())
            ppvs.append(ppv.item())
            recalls.append(recall.item())
        return np.asarray(dices), np.asarray(jscs), np.asarray(ppvs), np.asarray(recalls)


class RECALL_Average(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])

            recall = inter / (inter + torch.sum(logits[:, 0, :, :, :] * targets[:, 1, :, :, :]))
            dices.append(recall.item())
        return np.asarray(dices)