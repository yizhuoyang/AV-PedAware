import os
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from dataloader.data_processing import generate_music_gt
import io
from PIL import Image
gradients = {}

import torch
import torch.nn.functional as F

def masked_weighted_loss(predicted, target, mask, base_weight=1.0, masked_weight=200.0, loss_type="mse"):
    if loss_type == "mse":
        loss = F.mse_loss(predicted, target, reduction='none')
    elif loss_type == "l1":
        loss = F.l1_loss(predicted, target, reduction='none')
    else:
        raise ValueError("Unsupported loss type. Use 'mse' or 'l1'.")

    weight_map = base_weight + (mask * (masked_weight - base_weight))
    weighted_loss = loss * weight_map
    return weighted_loss.mean()

def _neg_loss(pred, gt):
    epsilon = 1e-7
    pred = torch.clamp(pred, epsilon, 1 - epsilon)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    num_pos = pos_inds.float().sum()

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 1) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 1) * neg_weights * neg_inds

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLossMUSIC(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        super(FocalLossMUSIC, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, spectrum, gt_spectrum):
        loss = (1 - spectrum) ** self.gamma * (gt_spectrum - spectrum) ** 2
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # 不聚合

def save_grad(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()  # 存储梯度
    return hook

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epoch, model_path, device="cuda", lr_scheduler=None, save_best=True,multi_task=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epoch  = epoch
        self.model_path = model_path
        self.save_best = save_best
        self.multi_task = multi_task
        self.mse_loss = nn.MSELoss()
        # self.mse_loss = _neg_loss
        self.writer = SummaryWriter(log_dir=os.path.join(model_path, "logs"))

    def train(self):
        best_val_loss = float("inf")
        num_epochs = self.epoch

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1}/{num_epochs}]")

            train_loss, train_mae = self._train_one_epoch(epoch)
            val_loss, val_mae = self._evaluate(epoch)
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()
            print(
                f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.3f} deg | "
                f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.3f} deg"
            )
            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                # os.makedirs(self.model_path,exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.model_path, "model_best.pth"))
                print(f"Best model saved at {self.model_path}/model_best.pth")
            torch.save(self.model.state_dict(), os.path.join(self.model_path, "last.pth"))

    def _train_one_epoch(self,epoch):
        self.model.train()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        for inputs, targets,sv,correlation in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets,sv,correlation = inputs.to(self.device).float(), targets.to(self.device).float(),sv.to(self.device),correlation.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs,sv,correlation)
            if len(outputs)==2:
                if targets.shape[1]==1:
                    sigma = 10
                else:
                    sigma = 5
                spectrum_gt = generate_music_gt(targets,sigma=sigma)
                loss = self.mse_loss(outputs[1],spectrum_gt)
                total_mae += circular_mae_from_spectrum(outputs[1], targets).item()
            else:
                loss = self.criterion(outputs, targets)
            try:
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            except RuntimeError as e:
                print("Error when calculating noise space")
                continue

        avg_train_loss = total_loss / max(num_batches, 1)
        avg_train_mae = total_mae / max(num_batches, 1)
        self.writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        self.writer.add_scalar("MAE/Train", avg_train_mae, epoch)
        return avg_train_loss, avg_train_mae


    def _evaluate(self,epoch):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets,sv,correlation in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets,sv,correlation = inputs.to(self.device), targets.to(self.device),sv.to(self.device),correlation.to(self.device)
                outputs = self.model(inputs,sv,correlation)
                if len(outputs)==2:
                    if targets.shape[1]==1:
                        sigma = 10
                    else:
                        sigma = 5
                    spectrum_gt = generate_music_gt(targets,sigma=sigma)
                    loss = self.mse_loss(outputs[1],spectrum_gt)
                    total_mae += circular_mae_from_spectrum(outputs[1], targets).item()
                else:
                    loss = self.criterion(outputs, targets)
                    # spectrum_gt = generate_music_gt(targets)
                    # loss = self.mse_loss(outputs,spectrum_gt)
                total_loss += loss.item()
                num_batches += 1

        avg_val_loss = total_loss / max(num_batches, 1)
        avg_val_mae = total_mae / max(num_batches, 1)
        self.writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        self.writer.add_scalar("MAE/Validation", avg_val_mae, epoch)
        return avg_val_loss, avg_val_mae


def circular_mae_deg(predicted, target):
    diff = torch.deg2rad(predicted - target)
    wrapped = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.rad2deg(torch.abs(wrapped)).mean()


def circular_mae_from_spectrum(spectrum, target):
    predicted = torch.argmax(spectrum, dim=1, keepdim=True).to(target.dtype)
    return circular_mae_deg(predicted, target)

def targets_to_onehot(targets, num_classes):

    batch_size = len(targets)
    targets_flat = torch.cat([torch.tensor(t, dtype=torch.long) for t in targets])
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    batch_indices = torch.arange(batch_size).repeat_interleave(lengths)
    one_hot = torch.zeros((batch_size, num_classes), dtype=torch.float32)
    one_hot[batch_indices, targets_flat] = 1

    return one_hot

class ModelTrainer_pretrain:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epoch, model_path, device="cuda", lr_scheduler=None, save_best=True,multi_task=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.epoch  = epoch
        self.model_path = model_path
        self.save_best = save_best
        self.multi_task = multi_task
        self.mse_loss = _neg_loss

    def train(self):
        best_val_loss = float("inf")
        num_epochs = self.epoch

        for epoch in range(num_epochs):
            print(f"🔹 Epoch [{epoch+1}/{num_epochs}]")

            train_loss = self._train_one_epoch()
            val_loss= self._evaluate()
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if self.save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                # os.makedirs(self.model_path,exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.model_path,"best_model"))
                print(f"🔥 Best model saved at {self.model_path}_bestmodel")

            torch.save(self.model.state_dict(), os.path.join(self.model_path,"last_model"))
            print(f"Last model saved at {self.model_path}_lastmodel")

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0

        for inputs, targets,mask in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, targets,mask = inputs.to(self.device).float(), targets.to(self.device).float(), mask.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = masked_weighted_loss(outputs, targets,mask)
            # loss = self.mse_loss(outputs, targets)
            # print(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)


    def _evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets,mask in tqdm(self.val_loader, desc="Validating", leave=False):
                inputs, targets,mask = inputs.to(self.device), targets.to(self.device),mask.to(self.device).float()
                outputs = self.model(inputs)
                loss = masked_weighted_loss(outputs, targets,mask)
                total_loss += loss.item()

        val_loss = total_loss / len(self.val_loader)
        return val_loss
