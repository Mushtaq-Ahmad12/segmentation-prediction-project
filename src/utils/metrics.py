# src/utils/metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LossFunctions:
    """Collection of loss functions for segmentation"""
    
    @staticmethod
    def dice_loss(pred, target, smooth=1.0):
        """Dice loss"""
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    @staticmethod
    def bce_loss(pred, target):
        """Binary Cross Entropy loss"""
        return F.binary_cross_entropy_with_logits(pred, target)
    
    @staticmethod
    def bce_dice_loss(pred, target, alpha=0.5):
        """Combined BCE and Dice loss"""
        bce = LossFunctions.bce_loss(pred, target)
        dice = LossFunctions.dice_loss(pred, target)
        return alpha * bce + (1 - alpha) * dice


class SegmentationMetrics:
    """Metrics for segmentation evaluation"""
    
    @staticmethod
    def iou_score(pred, target, threshold=0.5, smooth=1e-6):
        """Intersection over Union"""
        pred = (torch.sigmoid(pred) > threshold).float()
        
        # Flatten tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def dice_score(pred, target, threshold=0.5, smooth=1e-6):
        """Dice coefficient"""
        pred = (torch.sigmoid(pred) > threshold).float()
        
        # Flatten tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return dice.item()
    
    @staticmethod
    def accuracy_score(pred, target, threshold=0.5, smooth=1e-6):
        """Pixel-wise accuracy"""
        pred = (torch.sigmoid(pred) > threshold).float()
        
        # Flatten tensors
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        
        correct = (pred_flat == target_flat).sum()
        total = pred_flat.numel()
        
        accuracy = correct.float() / total
        return accuracy.item()
    
    @staticmethod
    def calculate_batch_metrics(target, pred, threshold=0.5):
        """Calculate all metrics for a batch"""
        metrics = {}
        
        with torch.no_grad():
            metrics['iou'] = SegmentationMetrics.iou_score(pred, target, threshold)
            metrics['dice'] = SegmentationMetrics.dice_score(pred, target, threshold)
            metrics['f1'] = metrics['dice']  # F1 = Dice for binary
            metrics['accuracy'] = SegmentationMetrics.accuracy_score(pred, target, threshold)
        
        return metrics