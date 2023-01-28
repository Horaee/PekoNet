from torch.autograd import Function


class GRL(Function):
    def forward(self, tensor):
        return tensor


    def backward(self, grad):
        return grad.neg()