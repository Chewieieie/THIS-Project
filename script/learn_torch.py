import torch
from torch import nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(20, 30),
            nn.ReLU(),
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.Linear(10, 2)
        )

    def forward(self, x):
        y = self.block(x)
        return y


class Temp:

    def __call__(self, *args, **kwargs):
        x = args[0]
        print(x)

    def add(self, x, y, z):
        print(x + y + z)

    def add2(self, *args, **kwargs):

        print('args is: {}'.format(args))
        print('kwargs is: {}'.format(kwargs))

        sum = 0
        for item in args:
            sum += item
        return sum


if __name__ == '__main__':
    pass