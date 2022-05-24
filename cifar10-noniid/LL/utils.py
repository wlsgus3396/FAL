from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
def imagenet_transformer():
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def cifar_transformer():
    return torchvision.transforms.Compose([
           torchvision.transforms.RandomHorizontalFlip(),
           torchvision.transforms.ToTensor(),
       ])
class MarginRankingLoss_learning_loss(nn.Module):
    def __init__(self, margin=1.0):
        super(MarginRankingLoss_learning_loss, self).__init__()
        self.margin = margin
    def forward(self, inputs, targets):
        random = torch.randperm(inputs.size(0))
        pred_loss = inputs[random]
        pred_lossi = pred_loss[:inputs.size(0)//2]
        pred_lossj = pred_loss[inputs.size(0)//2:]
        target_loss = targets.reshape(inputs.size(0), 1)
        target_loss = target_loss[random]
        target_lossi = target_loss[:inputs.size(0)//2]
        target_lossj = target_loss[inputs.size(0)//2:]
        final_target = torch.sign(target_lossi - target_lossj)
        
        return F.margin_ranking_loss(pred_lossi, pred_lossj, final_target, margin=self.margin, reduction='mean')