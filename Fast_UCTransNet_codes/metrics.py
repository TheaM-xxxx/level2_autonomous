# -*- coding: utf-8 -*-
# @Author   : Xiyao Ma

import torch
from torch import Tensor
import numpy as np

def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()

    inter = (input * target).sum()
    sets_sum = input.sum() + target.sum()
    # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (2 * inter + epsilon) / (sets_sum + epsilon)

    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), epsilon)

def multiclass_iou_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return iou_coeff(input.flatten(0, 1), target.flatten(0, 1), epsilon)


def iou_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()

    inter = (input * target).sum()
    sets_sum = input.sum() + target.sum()
    # sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    iou = (inter + epsilon) / (sets_sum - inter + epsilon)

    return iou.mean()

def numeric_score(input: Tensor, target: Tensor):
    """Computes scores:
    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives
    return: FP, FN, TP, TN"""

    FP = torch.sum((input == 1) & (target == 0)).item()
    FN = torch.sum((input == 0) & (target == 1)).item()
    TP = torch.sum((input == 1) & (target == 1)).item()
    TN = torch.sum((input == 0) & (target == 0)).item()
    # print(FP, FN, TP, TN)
    return FP, FN, TP, TN

def classify_scores(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """Getting the accuracy of the model"""
    FP, FN, TP, TN = numeric_score(input, target)
    accuracy = (TP + TN + epsilon) / (FP + FN + TP + TN + epsilon)
    precision = (TP + epsilon) / (FP + TP + epsilon)
    recall = (TP + epsilon) / (FN + TP + epsilon)
    return accuracy,precision,recall


# def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
#     # Dice loss (objective to minimize) between 0 and 1
#     fn = multiclass_dice_coeff if multiclass else dice_coeff
#     return 1 - fn(input, target, reduce_batch_first=True)
