import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable

__all__ = ['ensembleNovel','ensembleMLP','flannel']

class ENet(nn.Module):
    def __init__(self):
        super(ENet, self).__init__()
#        self.w1 = nn.Linear(5, 1, bias=True) 
#        self.w2 = nn.Linear(5, 1, bias=True) 
#        self.w3 = nn.Linear(5, 1, bias=True) 
#        self.w4 = nn.Linear(5, 1, bias=True)
#        self.vectors = Variable(torch.randn(1, 5), requires_grad=True).unsqueeze(-1)
        self.ew1 = nn.Linear(20, 4, bias=True)
        self.ew2 = nn.Linear(128, 4, bias=True)
#        self.act = nn.Dropout(p=0.5)
    def forward(self, x):
        v = self.ew1(x)
#        v = self.ew2(self.ew1(x))
        return v

class GENet(nn.Module):
    def __init__(self):
        super(GENet, self).__init__()
        self.w1 = nn.Linear(20, 1, bias=True) 
        self.w2 = nn.Linear(20, 1, bias=True) 
        self.w3 = nn.Linear(20, 1, bias=True) 
        self.w4 = nn.Linear(20, 1, bias=True)
        self.w5 = nn.Linear(20, 1, bias=True)
        
        self.vectors = Variable(torch.FloatTensor([[1.]]))
#        self.w_vec = nn.Linear(1, 5, bias=True)
        self.ew1 = nn.Linear(400, 5, bias=True)
        self.ew2 = nn.Linear(16, 4, bias=True)
        self.ew3 = nn.Linear(420, 128, bias=True)
        self.ew4 = nn.Linear(20, 4, bias=True)
        self.act = nn.Tanh()
#        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        zzz = x.split([4,4,4,4,4],1)
        z = torch.stack(zzz, 1)
        L = x.shape[0]
        x1 = x.unsqueeze(-1)
        x2 = x1.permute(0,2,1)
        x3 = torch.matmul(x1,x2)
        x3 = x3.view(L, -1)
        wv = self.act(self.ew1(x3)).unsqueeze(-1)
#        print (wv.shape, z.shape)
        v = torch.sum(wv * z, 1)
        return v

class FlannelNet(nn.Module):
    def __init__(self):
        super(FlannelNet, self).__init__()
        self.w1 = nn.Linear(20, 1, bias=True) 
        self.w2 = nn.Linear(20, 1, bias=True) 
        self.w3 = nn.Linear(20, 1, bias=True) 
        self.w4 = nn.Linear(20, 1, bias=True)
        self.w5 = nn.Linear(20, 1, bias=True)
        
        self.vectors = Variable(torch.FloatTensor([[1.]]))
#        self.w_vec = nn.Linear(1, 5, bias=True)
        self.ew1 = nn.Linear(400, 5, bias=True)
        self.ew2 = nn.Linear(16, 4, bias=True)
        self.ew3 = nn.Linear(420, 128, bias=True)
        self.ew4 = nn.Linear(20, 4, bias=True)
        self.act = nn.Tanh()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        zzz = x.split([4,4,4,4,4],1)
        z = torch.stack(zzz, 1)
        L = x.shape[0]
        x1 = x.unsqueeze(-1)
        x2 = x1.permute(0,2,1)
        x3 = torch.matmul(x1,x2)
        x3 = x3.view(L, -1)
        wv = self.sm(self.ew1(x3)).unsqueeze(-1)
        v = torch.sum(wv * z, 1)
        return v

def flannel(pretrained=False, progress=True, **kwargs):
    model = FlannelNet(**kwargs)
    return model

def ensembleNovel(pretrained=False, progress=True, **kwargs):
    model = GENet(**kwargs)
    return model

def ensembleMLP(pretrained=False, progress=True, **kwargs):
    model = ENet(**kwargs)
    return model
	
if __name__ == '__main__':
    enet = flannel(pretrained=True).double()
    data = torch.from_numpy(np.random.rand(3,20))
#    print (data)
    y = enet(data)
    print (y)

