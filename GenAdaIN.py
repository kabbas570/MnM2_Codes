import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x, (2, 3)) / (x.shape[2] * x.shape[3])

    def sigma(self, x):
        """Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""

        return torch.sqrt(
            (
                torch.sum(
                    (x.permute([2, 3, 0, 1]) - self.mu(x)).permute([2, 3, 0, 1]) ** 2,
                    (2, 3),
                )
                + 0.000000023
            )
            / (x.shape[2] * x.shape[3])
        )

    def forward(self, x, y):
        """Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""
        return (
            self.sigma(y) * ((x.permute([2, 3, 0, 1]) - self.mu(x)) / self.sigma(x))
            + self.mu(y)
        ).permute([2, 3, 0, 1])


class GenAdaIN(nn.Module):
    """
    Adaptive Instance Normalization with generalization to higher-dimensional data (Nd)

    Implements Equation 8 of the paper

    Input can be 1d, 2d, 3d or nd (where n>=1)

    Author: Muhammad Asad (masadcv@gmail.com)
    """

    def __init__(self):
        super().__init__()

    def get_reduce_dims(self, xshape):
        return [i for i, _ in enumerate(xshape) if i > 1]

    def mu(self, x):
        reduce_dims = self.get_reduce_dims(x.shape)
        return torch.mean(x, dim=reduce_dims, keepdim=True)

    def sigma(self, x):
        reduce_dims = self.get_reduce_dims(x.shape)
        return torch.std(x, dim=reduce_dims, keepdims=True)

    def forward(self, x, y):
        eps = torch.finfo(torch.float32).eps
        return self.sigma(y) * ((x - self.mu(x)) / (self.sigma(x) + eps)) + self.mu(y)


if __name__ == "__main__":
    # Test 1: check if GenAdaIN matches AdaIN in 2d case
    tensorshape_2d = [
        2,
        1,
    ] + [16] * 2
    tensor1_2d = torch.rand(size=tensorshape_2d)
    tensor2_2d = torch.rand(size=tensorshape_2d)

    tensorout_gadain = GenAdaIN()(tensor1_2d, tensor2_2d)
    tensorout_adain = AdaIN()(tensor1_2d, tensor2_2d)

    # should be close to zero for a match
    print(torch.sum(torch.abs(tensorout_adain - tensorout_gadain)))

    # Test 2: check if GenAdaIN works for 3d case
    tensorshape_3d = [
        2,
        1,
    ] + [16] * 3
    tensor1_3d = torch.rand(size=tensorshape_3d)
    tensor2_3d = torch.rand(size=tensorshape_3d)

    # should work
    tensorout_gadain = GenAdaIN()(tensor1_3d, tensor2_3d)

    print(tensorout_gadain.shape)
    print(tensor1_3d.shape)

    # Test 3: check if GenAdaIN works for nd case (where n = 4 or beyond)
    tensorshape_4d = [
        2,
        1,
    ] + [16] * 4
    tensor1_4d = torch.rand(size=tensorshape_4d)
    tensor2_4d = torch.rand(size=tensorshape_4d)

    # should work
    tensorout_gadain = GenAdaIN()(tensor1_4d, tensor2_4d)

    print(tensorout_gadain.shape)
    print(tensor1_4d.shape)
