import torch
import torchvision.models as models
from torch import nn
from text_model import TextPreprocessor, load_tweet_text_data, load_image_text_data


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super(InceptionV3FeatureExtractor, self).__init__()
        self.inception_v3 = models.inception_v3(pretrained=True)

        self.flatten = nn.Flatten()

    def forward(self, x):
        # 遍历
        for name, module in self.inception_v3.named_children():
            if name == 'AuxLogits':
                continue
            x = module(x)
            if name == 'avgpool':
                return self.flatten(x)

        return x


if __name__ == '__main__':
    # inception_v3 = models.inception_v3(pretrained=True)
    # print(inception_v3)

    fl = nn.Flatten(start_dim=1, end_dim=2)

    tensor = torch.randn(1, 1, 2048)

    out = fl(tensor)
    print(out.shape)


    # model = InceptionV3FeatureExtractor()
    #
    # input_tensor = torch.randn(2, 3, 512, 512) # (batch_size, channel, height, width)
    # out = model(input_tensor) # __call__
    #
    # print(out.shape)

    # print(out.shape)
    # model = LSTMNet()
    pass

