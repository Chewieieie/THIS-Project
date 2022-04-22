from image_model import InceptionV3FeatureExtractor
from text_model import LSTMNet, tweet_text_data_dict, image_text_data_dict, text_preprocessor
import torch
from torch import nn
from torch.utils import data
import os
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import random


class THISDataset(data.Dataset):
    def __init__(self, image_type, tweet_text_dict=tweet_text_data_dict, image_text_dict=image_text_data_dict, mode='train'):
        """
        :param image_type: the image type, 0 - NotHate, 1 - Racist, 2 - Sexist, 3 - Homophobe, 4 - Religion, 5 - OtherHate.
        :param tweet_text_dict: tweet text dict in the dataset, key is the tweet id, value is the word list of that text.
        :param image_text_dict: image text dict in the dataset, key is the tweet id, value is the word list of that text.
        """
        negative_type_list = []
        for i in range(6):
            if i == image_type:
                continue
            negative_type_list.append(i)

        self.image_base_dir = '../dataset/' + mode + '/' + str(image_type) + '/'
        self.image_data_list = os.listdir(self.image_base_dir)

        self.negative_base_dir_list = []
        self.negative_data_list = []
        for negative_type in negative_type_list:
            base_dir = '../dataset/' + mode + '/' + str(negative_type) + '/'
            self.negative_base_dir_list.append(base_dir)
            self.negative_data_list.append(os.listdir(base_dir))

        # padded sentence length
        self.sen_len = text_preprocessor.get_sen_len()

        # dataset transform
        self.transform = transforms.Compose([transforms.Resize(224),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                                             ])

        self.tweet_text_dict = text_preprocessor.get_sentence_word2idx_dict(tweet_text_dict)
        self.image_text_dict = text_preprocessor.get_sentence_word2idx_dict(image_text_dict)

    def __getitem__(self, idx):
        # read image data
        image_file_name = self.image_data_list[idx]
        image_file_path = self.image_base_dir + image_file_name

        image = Image.open(image_file_path).convert('RGB')
        image = self.transform(image)

        # tweet id
        tweet_id = str(image_file_name).split('.')[0]

        # read tweet text embedding
        tweet_text = self.tweet_text_dict.get(tweet_id, None)

        # read image text embedding
        image_text = self.image_text_dict.get(tweet_id, None)

        if image_text is None:
            image_text = torch.zeros(self.sen_len)

        positive_data = (image, tweet_text, image_text)

        # random select a negative data
        negative_type = random.randint(0, 4)
        negative_base_dir = self.negative_base_dir_list[negative_type]
        negative_data_list = self.negative_data_list[negative_type]
        negative_image_file_name = negative_data_list[random.randint(0, len(negative_data_list)-1)]
        negative_image_file_path = negative_base_dir + negative_image_file_name

        n_image = self.transform(Image.open(negative_image_file_path).convert('RGB'))
        n_tweet_id = str(negative_image_file_name).split('.')[0]
        # read tweet text embedding
        n_tweet_text = self.tweet_text_dict.get(n_tweet_id, None)

        # read image text embedding
        n_image_text = self.image_text_dict.get(n_tweet_id, None)

        if n_image_text is None:
            n_image_text = torch.zeros(self.sen_len, dtype=torch.long)

        negative_data = (n_image, n_tweet_text, n_image_text)

        return (positive_data, negative_data)

    def __len__(self):
        return len(self.image_data_list)


class FCMModel(nn.Module):
    def __init__(self):
        super(FCMModel, self).__init__()
        # inception v3 image model
        self.inception_v3 = InceptionV3FeatureExtractor()

        # lstm model
        self.lstm = LSTMNet()

        # fc model
        self.fc = nn.Sequential(
            nn.Linear(2048 + 150 + 150, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, image_text, tweet_text):
        # image feature (bs, 2048)
        img_feat = self.inception_v3(img)

        # detach from inception v3 model
        img_feat = img_feat.detach()

        # image_text feature (bs, 300)
        text_input = torch.cat([image_text, tweet_text], dim=0)
        text_feat = self.lstm(text_input)

        # fusion feature (bs, 2348)
        fuse_feat = torch.cat([img_feat, text_feat], dim=-1)

        # final prediction
        out = self.fc(fuse_feat)

        return out


def eval_FCM(image_type=0):
    # evaluation parameters config
    use_gpu = torch.cuda.is_available()
    batch_size = 10

    acc_matrix = [[0] * 5 for _ in range(6)]
    for i in range(6):
        fcm_model = torch.load('../save/fcm_{}_epoch_50.pkl')

        for k in range(6):
            ds = THISDataset(image_type=k, mode='test')
            train_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=0,
                                      drop_last=True)

            total_cnt = 0
            right = 0
            fault = 0

            for j, data in enumerate(train_loader):
                p_data = data[0]
                p_image = p_data[0]
                p_tweet_text = p_data[1].long()
                p_image_text = p_data[2].long()

                if use_gpu:
                    p_image = p_image.cuda()
                    p_tweet_text = p_tweet_text.cuda()
                    p_image_text = p_image_text.cuda()

                pred = fcm_model.forward(p_image, p_tweet_text, p_image_text)
                pred = torch.squeeze(pred)
                pred_list = pred.tolist()

                if j == image_type:
                    for res in pred_list:
                        total_cnt += 1
                        if res >= 0.5:
                            right += 1
                        else:
                            fault += 1
                else:
                    for res in pred_list:
                        total_cnt += 1
                        if res < 0.5:
                            right += 1
                        else:
                            fault += 1

            acc_matrix[i][k] = right / total_cnt

    return acc_matrix


def train_FCM(image_type=0):
    # training parameters config
    use_gpu = torch.cuda.is_available()
    train_epoch = 100
    batch_size = 64
    print_loss_every_batch = 20
    save_weight_every_epoch = 5

    loss_func = nn.BCELoss()
    if use_gpu:
        loss_func = loss_func.cuda()

    fcm_model = FCMModel()
    if use_gpu:
        fcm_model = fcm_model.cuda()

    optimizer = torch.optim.Adam(fcm_model.parameters(), lr=0.001)
    THIS_dataset = THISDataset(image_type)
    train_loader = DataLoader(dataset=THIS_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    for epoch in range(train_epoch):
        print('--------------start training epoch {}--------------'.format(epoch))
        for i, data in enumerate(train_loader):
            # train one step on positive data
            p_data = data[0]
            p_image = p_data[0]
            p_tweet_text = p_data[1].long()
            p_image_text = p_data[2].long()

            if use_gpu:
                p_image = p_image.cuda()
                p_tweet_text = p_tweet_text.cuda()
                p_image_text = p_image_text.cuda()

            pred = fcm_model.forward(p_image, p_tweet_text, p_image_text)
            pred = torch.squeeze(pred)
            target = torch.ones(batch_size, dtype=torch.float)

            if use_gpu:
                target = target.cuda()

            loss = loss_func(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # train one step on negative data
            n_data = data[1]
            n_image = n_data[0]
            n_tweet_text = n_data[1].long()
            n_image_text = n_data[2].long()

            if use_gpu:
                n_image = n_image.cuda()
                n_tweet_text = n_tweet_text.cuda()
                n_image_text = n_image_text.cuda()

            pred = fcm_model.forward(n_image, n_tweet_text, n_image_text)
            pred = torch.squeeze(pred)
            target = torch.zeros(batch_size, dtype=torch.float)

            if use_gpu:
                target = target.cuda()

            loss2 = loss_func(pred, target)
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

            if i % print_loss_every_batch == 0:
                print('[epoch #{}] training loss on batch #{}: {}'.format(epoch, i, (loss2.item() + loss.item()) / 2))

        if (epoch+1) % save_weight_every_epoch == 0:
            torch.save(fcm_model, '../save/fcm_{}_epoch_{}.pkl'.format(image_type, epoch+1))


if __name__ == '__main__':
    train_FCM(1)





