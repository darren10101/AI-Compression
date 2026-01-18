import torch
import torch.nn as nn
import torch.nn.functional as F

class RoundSTE(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input):
    return torch.round(input)
  @staticmethod
  def backward(ctx, grad_output):
    return grad_output

def round_ste(x):
  return RoundSTE.apply(x)

