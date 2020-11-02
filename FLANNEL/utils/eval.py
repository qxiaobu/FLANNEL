from __future__ import print_function, absolute_import

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k / batch_size)
    return res

def bce_accuracy(ppp, rrr):
    batch_size = len(ppp)
    y = 0.
    for i in range(batch_size):
      if ppp[i] >= 0.5 and rrr[i] == 1.:
        y += 1.
      if ppp[i] < 0.5 and rrr[i] == 0.:
        y += 1.
    return float(y/batch_size)