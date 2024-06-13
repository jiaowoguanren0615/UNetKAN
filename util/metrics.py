import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

from typing import Tuple

from .utils import get_world_size


class Metrics:
    def __init__(self, num_classes: int, ignore_label: int, device) -> None:
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.hist = torch.zeros(num_classes, num_classes).to(device)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.argmax(dim=1).flatten()
        keep = target != self.ignore_label
        self.hist += torch.bincount(target[keep] * self.num_classes + pred[keep], minlength=self.num_classes**2).view(self.num_classes, self.num_classes)

    def compute_iou(self) -> Tuple[Tensor, Tensor]:
        ious = self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1) - self.hist.diag())
        miou = ious[~ious.isnan()].mean().item()
        ious *= 100
        miou *= 100
        return ious.cpu().numpy().round(2).tolist(), round(miou, 2)

    def compute_f1(self) -> Tuple[Tensor, Tensor]:
        f1 = 2 * self.hist.diag() / (self.hist.sum(0) + self.hist.sum(1))
        mf1 = f1[~f1.isnan()].mean().item()
        f1 *= 100
        mf1 *= 100
        return f1.cpu().numpy().round(2).tolist(), round(mf1, 2)

    def compute_pixel_acc(self) -> Tuple[Tensor, Tensor]:
        acc = self.hist.diag() / self.hist.sum(1)
        macc = acc[~acc.isnan()].mean().item()
        acc *= 100
        macc *= 100
        return acc.cpu().numpy().round(2).tolist(), round(macc, 2)

    def reduce_from_all_processes(self):
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        dist.barrier()
        dist.all_reduce(self.hist)


def all_gather(data):
    """
    收集各个进程中的数据
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()  # 进程数
    if world_size == 1:
        return [data]

    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)

    return data_list


class MeanAbsoluteError(object):
    def __init__(self):
        self.mae_list = []

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        error_pixels = torch.sum(torch.abs(resize_pred - gt), dim=(1, 2, 3)) / (h * w)
        self.mae_list.extend(error_pixels.tolist())

    def compute(self):
        mae = sum(self.mae_list) / len(self.mae_list)
        return mae

    def gather_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        gather_mae_list = []
        for i in all_gather(self.mae_list):
            gather_mae_list.extend(i)
        self.mae_list = gather_mae_list

    def __str__(self):
        mae = self.compute()
        return f'MAE: {mae:.3f}'


class F1Score(object):
    """
    refer: https://github.com/xuebinqin/DIS/blob/main/IS-Net/basics.py
    """

    def __init__(self, threshold: float = 0.5):
        self.precision_cum = None
        self.recall_cum = None
        self.num_cum = None
        self.threshold = threshold

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        batch_size, c, h, w = gt.shape
        assert batch_size == 1, f"validation mode batch_size must be 1, but got batch_size: {batch_size}."
        resize_pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
        gt_num = torch.sum(torch.gt(gt, self.threshold).float())

        pp = resize_pred[torch.gt(gt, self.threshold)]  # 对应预测map中GT为前景的区域
        nn = resize_pred[torch.le(gt, self.threshold)]  # 对应预测map中GT为背景的区域

        pp_hist = torch.histc(pp, bins=255, min=0.0, max=1.0)
        nn_hist = torch.histc(nn, bins=255, min=0.0, max=1.0)

        # Sort according to the prediction probability from large to small
        pp_hist_flip = torch.flipud(pp_hist)
        nn_hist_flip = torch.flipud(nn_hist)

        pp_hist_flip_cum = torch.cumsum(pp_hist_flip, dim=0)
        nn_hist_flip_cum = torch.cumsum(nn_hist_flip, dim=0)

        precision = pp_hist_flip_cum / (pp_hist_flip_cum + nn_hist_flip_cum + 1e-4)
        recall = pp_hist_flip_cum / (gt_num + 1e-4)

        if self.precision_cum is None:
            self.precision_cum = torch.full_like(precision, fill_value=0.)

        if self.recall_cum is None:
            self.recall_cum = torch.full_like(recall, fill_value=0.)

        if self.num_cum is None:
            self.num_cum = torch.zeros([1], dtype=gt.dtype, device=gt.device)

        self.precision_cum += precision
        self.recall_cum += recall
        self.num_cum += batch_size

    def compute(self):
        pre_mean = self.precision_cum / self.num_cum
        rec_mean = self.recall_cum / self.num_cum
        f1_mean = (1 + 0.3) * pre_mean * rec_mean / (0.3 * pre_mean + rec_mean + 1e-8)
        max_f1 = torch.amax(f1_mean).item()
        return max_f1

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.precision_cum)
        torch.distributed.all_reduce(self.recall_cum)
        torch.distributed.all_reduce(self.num_cum)

    def __str__(self):
        max_f1 = self.compute()
        return f'maxF1: {max_f1:.3f}'