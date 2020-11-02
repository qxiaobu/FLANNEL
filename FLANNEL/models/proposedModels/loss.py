import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, label_distri = None, size_average=True, model_name = None, cuda_a = False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.label_distri = label_distri
        self.model_name = model_name
        self.cuda_a = cuda_a

    def forward(self, input_d, target_d):
        if self.model_name == None or 'foda' in self.model_name:
          p = input_d
        else:
#          p=F.sigmoid(input_d)
          p=F.softmax(input_d,dim=1)
#          p = input_d

#        temp = torch.zeros(p.shape,dtype=torch.float)
#        if self.cuda_a:
#          temp = temp.cuda()
#        target_v=temp.scatter_(1,torch.unsqueeze(target_d,dim=1),1.)
        log_p=torch.log(p + 1e-9)
        temp = torch.zeros(p.shape)
        if self.cuda_a:
          temp = temp.cuda()
        target_v=temp.scatter_(1,torch.unsqueeze(target_d,dim=1),1.)
        
#        loss = torch.abs(p - target_v)
#        loss = -1 * (target_v * torch.log(p + 1e-9) + (1 - target_v) * torch.log(1 - p + 1e-9))
        
        if self.label_distri != None:
          loss = -1 * (1-p)**self.gamma * log_p * target_v * self.label_distri
        else:  
          loss = -1 * (1-p)**self.gamma * log_p * target_v
        if self.size_average: 
          return loss.sum()
        else: 
          return loss.sum()

				
class MultiLoss(nn.Module):
  
    def __init__(self):
        super(MultiLoss, self).__init__()
        self.cuda_a = cuda_a

    def forward(self, input_d, target_d):
        p=F.sigmoid(input_d)
        log_p=torch.log(p + 1e-9)
        log_np = torch.log(1.- p + 1e-9)
        temp = torch.zeros(p.shape)
        temp = temp.cuda()
        target_v=temp.scatter_(1,torch.unsqueeze(target_d,dim=1),1.)
        
        loss = -1 * [target_v * log_p + (1 - target_v) * log_np]
        
        return loss.sum()

class AutoLoss(nn.Module):
    def __init__(self):
        super(AutoLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.l1_loss = nn.L1Loss(reduce=False, size_average=False)

    def forward(self, input_d, target_d, input_x, target_x):
        loss_label = self.ce_loss(input_d, target_d)
        loss_x = self.l1_loss(input_x, target_x)
#        print (loss_label)
#        print (loss_x)
        l1 = loss_label.sum()
        l2 = loss_x.mean()
        return l1 + 0.1 * l2

class MultiLoss(nn.Module):
    def __init__(self):
        super(MultiLoss, self).__init__()

    def forward(self, input_d, target_d):
        p=F.sigmoid(input_d)
        temp = torch.zeros(p.shape).cuda()
        target_v=temp.scatter_(1,torch.unsqueeze(target_d,dim=1),1.)
        
        log_p=torch.log(p + 1e-9)
        log__p=torch.log(1 - p + 1e-9)
        loss = -1 * (target_v * log_p + (1 - target_v) * log__p)
        return loss.sum()

class PairwiseLoss(nn.Module):
    def __init__(self):
        super(PairwiseLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.be_loss = nn.BCELoss()
        
    def forward(self, input1=None, input2=None, input3=None, target_1=None, target_2=None, label_d=None, ttt=None):
        if ttt == 'train':
          l1 = self.ce_loss(input1, target_1)
          l2 = self.ce_loss(input2, target_2)
          l3 = self.be_loss(input3, label_d)
          return l1+l2+l3
        else:
          l1 = self.ce_loss(input1, target_1)
          return l1
        