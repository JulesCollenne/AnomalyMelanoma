import datetime
import math
import shutil
import time
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import CustomDataset
from training_setups.utils import AverageMeter, ProgressMeter, save_checkpoint


class TrainingSetup(nn.Module):
    def __init__(self, base_model):
        super(TrainingSetup, self).__init__()
        self.base_model = base_model
        self.dataset = CustomDataset("/media/jules/Transcend/Datasets/isic/ISIC_2020/GroundTruth.csv",
                                     transform=None)
        self.epochs = 100
        self.batch_size = 16
        self.current_epoch = 0
        self.distributed = True
        self.gpu = 0
        self.print_freq = 10
        self.init_lr = 0.05
        self.multiprocessing_distributed = True

        self.rank = 0
        self.ngpus_per_node = 0

        self.UA = False
        self.optimizer = None

        self.path_name = lambda: join("checkpoints", f"{self.name}_{self.base_model.name}_{self.current_epoch}")

    def kfold_training(self, kf):
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            print(f"Training fold {fold+1}...")
            train_data = torch.utils.data.Subset(self.dataset, train_idx)
            val_data = torch.utils.data.Subset(self.dataset, val_idx)

            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.batch_size)

            self.optimizer = torch.optim.SGD(self.encoder.parameters(), lr=self.init_lr,
                                             momentum=0.9, weight_decay=1e-6)
            criterion = nn.CosineEmbeddingLoss()
            losses = []
            old_loss = 999.

            if self.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
            else:
                self.train_sampler = None

            exit()

            for epoch in range(self.epochs):
                self.current_epoch = epoch
                if self.distributed:
                    train_sampler.set_epoch(epoch)

                self.adjust_learning_rate()
                current_loss = self.train_one_epoch(train_loader, criterion)
                losses.append(current_loss)

                if not self.multiprocessing_distributed or (self.multiprocessing_distributed
                                                            and self.rank % self.ngpus_per_node == 0):
                    if current_loss < old_loss:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': self.name,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                        }, is_best=False, filename=self.path_name(), UA=self.UA)
                        old_loss = current_loss

                # Validation loop here
                self.base_model.eval()
                with torch.no_grad():
                    # Implement your validation logic
                    pass

            # Save the trained model
            model_path = f"fold-{fold}.pth"
            torch.save(self.base_model.state_dict(), model_path)
            print(f"Saved model for fold {fold} as {model_path}")

    def train_one_epoch(self, train_loader, criterion):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4f')
        loss = None
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(self.current_epoch))

        la_queue = []
        qsize = 100
        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (images, _) in enumerate(train_loader):
            data_time.update(time.time() - end)
            if self.gpu is not None:
                images[0] = images[0].cuda(self.gpu, non_blocking=True)
                images[1] = images[1].cuda(self.gpu, non_blocking=True)

            # compute output and loss
            p1, p2, z1, z2 = self.model(x1=images[0], x2=images[1])
            # la_queue.append(z1.T)
            # la_queue.append(z2.T)
            # if len(la_queue) > qsize:
            #     la_queue[0] = la_queue.pop(-1)
            #     la_queue[1] = la_queue.pop(-1)
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            # if args.UA:
            #     loss = add_unif_align(p1, p2, z1, z2, loss_constr, align_alpha=0.5)
            # else:
            #     loss = loss_constr

            losses.update(loss.item(), images[0].size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                progress.display(i)

            return losses.avg

    def adjust_learning_rate(self):
        """Decay the learning rate based on schedule"""
        cur_lr = self.init_lr * 0.5 * (1. + math.cos(math.pi * self.current_epoch / self.epochs))
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = self.init_lr
            else:
                param_group['lr'] = cur_lr