import torch
from torch import Tensor

def dice_coeff_list(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    n_batch, n_class, n_w, n_h = input.size()
    if input.dim() == 2 or reduce_batch_first:
        x1 = input.reshape(n_batch, -1)
        x2 = target.reshape(n_batch, -1)
        # print ('x1', x1.shape, 'x2', x2.shape)
        sum_x1 = torch.sum(x1, -1)
        sum_x2 = torch.sum(x2, -1)
        sum_inter = torch.sum(x1 * x2, -1)
        # print ('sum_x1', sum_x1.shape, 'sum_x2', sum_x2.shape, 'sum_inter', sum_inter.shape)
        dice = (2 * sum_inter + epsilon) / (sum_x1 + sum_x2)

        # inter = torch.dot(input.reshape(-1), target.reshape(-1))
        # sets_sum = torch.sum(input) + torch.sum(target)
        # if sets_sum.item() == 0:
        #     sets_sum = 2 * inter
        # dice = (2 * inter + epsilon) / (sets_sum + epsilon)
        return dice

    else:
        # compute and average metric for each batch element
        dice = []
        for i in range(input.shape[0]):
            dice.append(dice_coeff(input[i, ...], target[i, ...]))
        return dice

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def dice_loss_list(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()

    fn = multiclass_dice_coeff if multiclass else dice_coeff_list
    return 1 - fn(input, target, reduce_batch_first=True)
