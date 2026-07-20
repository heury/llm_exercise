import torch


class AddGaussianNoise:
    """
    Adds Gaussian noise to a tensor.
    """
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ClampTransform:
    def __init__(self, min_val=0., max_val=1.):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        return tensor.clamp(self.min_val, self.max_val)
