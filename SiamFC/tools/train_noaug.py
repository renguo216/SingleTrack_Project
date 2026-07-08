from __future__ import absolute_import

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from got10k.datasets import *

from siamfc import TrackerSiamFC, ops
from siamfc.transforms import Compose, CenterCrop, ToTensor
from siamfc.datasets import Pair


class NoAugTransforms(object):
    """无数据增强的 transforms: 只用 CenterCrop + ToTensor"""
    def __init__(self, exemplar_sz=127, instance_sz=255, context=0.5):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context

        self.transforms_z = Compose([
            CenterCrop(exemplar_sz),
            ToTensor()])
        self.transforms_x = Compose([
            CenterCrop(instance_sz),
            ToTensor()])

    def __call__(self, z, x, box_z, box_x):
        z = self._crop(z, box_z, self.instance_sz)
        x = self._crop(x, box_x, self.instance_sz)
        z = self.transforms_z(z)
        x = self.transforms_x(x)
        return z, x

    def _crop(self, img, box, out_size):
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        center, target_sz = box[:2], box[2:]
        context = self.context * np.sum(target_sz)
        size = np.sqrt(np.prod(target_sz + context))
        size *= out_size / self.exemplar_sz
        avg_color = np.mean(img, axis=(0, 1), dtype=float)
        patch = ops.crop_and_resize(
            img, center, size, out_size,
            border_value=avg_color)
        return patch


class TrackerSiamFC_NoAug(TrackerSiamFC):
    """无数据增强的 SiamFC 训练器"""

    def __init__(self, **kwargs):
        super(TrackerSiamFC_NoAug, self).__init__(**kwargs)

    def train_over(self, seqs, val_seqs=None, save_dir='snapshot_noaug'):
        self.net.train()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        transforms = NoAugTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)

        dataset = Pair(seqs=seqs, transforms=transforms)

        # Windows 下 num_workers=0 避免多进程pickle问题
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True)

        # 训练日志
        log_file = os.path.join(save_dir, 'training_log_noaug.txt')
        with open(log_file, 'w') as f:
            f.write('Epoch,Iter,Loss\n')

        for epoch in range(self.cfg.epoch_num):
            epoch_losses = []
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                epoch_losses.append(loss)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            avg_loss = np.mean(epoch_losses)
            with open(log_file, 'a') as f:
                f.write('%d,%d,%.5f\n' % (epoch + 1, len(dataloader), avg_loss))

            self.lr_scheduler.step()

            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
            print('Epoch %d saved: %s (avg loss: %.5f)' % (epoch + 1, net_path, avg_loss))

        print('训练完成! 模型保存至: %s' % save_dir)
        print('训练日志保存至: %s' % log_file)


if __name__ == '__main__':
    root_dir = r'D:\experiment\SingleTrack_Project\data\datasets'
    seqs = GOT10k(root_dir, subset='train', return_meta=True)

    tracker = TrackerSiamFC_NoAug(
        epoch_num=50,
        batch_size=8,
        initial_lr=1e-2,
        ultimate_lr=1e-5,
        weight_decay=5e-4,
        momentum=0.9,
        num_workers=0)

    tracker.train_over(seqs, save_dir='snapshot_noaug')