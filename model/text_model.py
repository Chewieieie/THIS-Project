import json
import os
import re
import torch
from torch import nn
from gensim.models import word2vec, Word2Vec
from torch.utils import data
from torch.utils.data import DataLoader


class TextDataset(data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def load_image_text_data():
    token = re.compile('[A-Za-z]+|[!?,.]]') # 正则表达式
    base_dir = '../MMHS150K/img_txt/'
    json_file_list = os.listdir(base_dir)
    x = []
    y = {}
    for file_name in json_file_list:
        with open(base_dir + file_name) as file:
            d = json.loads(file.readline())
            image_text = str(d['img_text']).lower()
            image_text_word_list = token.findall(image_text)
            x.append(image_text_word_list)
            y[file_name.split('.')[0]] = image_text_word_list
    return x, y


def load_tweet_text_data():
    token = re.compile('[A-Za-z]+|[!?,.]]')

    with open('../MMHS150K/MMHS150K_GT.json') as file:
        d = json.loads(file.readline())

    x = []
    y = {}
    for tweet_id, val_dict in d.items():
        tweet_text = str(val_dict['tweet_text'])
        tweet_text = str(tweet_text).split("https")[0].lower()  # remove the http links
        tweet_text_word_list = token.findall(tweet_text)
        x.append(tweet_text_word_list)
        y[tweet_id] = tweet_text_word_list
    return x, y


def train_word2vec(x):
    model = word2vec.Word2Vec(x, sg=1)
    return model


def do_train():
    save_path = '../save/w2v.model'
    image_text = image_text_data_list
    tweet_text = tweet_text_data_list
    all_text = tweet_text + image_text

    print('all text length is: {}'.format(len(all_text)))

    word2vec_model = train_word2vec(all_text)
    word2vec_model.save(save_path)


def get_embedding_matrix():
    return embedding_matrix


class LSTMNet(nn.Module):
    def __init__(self, num_layers=2, hidden_dim=150, fix_embedding=True):
        super(LSTMNet, self).__init__()
        embedding = get_embedding_matrix()
        # embedding layer define
        self.embedding_dim = embedding.size(1)
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False if fix_embedding else True

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x, None)
        x = x[:, -1, :]  # final hidden state
        bs = x.size(0)
        x = torch.reshape(x, (int(bs / 2), -1))
        return x


def LSTMNet_test():
    sen_len = 30
    train_x = tweet_text_data_list
    text_preprocess = TextPreprocessor(train_x, sen_len)
    embedding_matrix = text_preprocess.make_embedding()

    print('embedding_matrix: {}'.format(embedding_matrix.shape))

    train_x = text_preprocess.sentence_word2idx()
    train_dataset = TextDataset(data=train_x, label=None)
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=2)

    model = LSTMNet()

    for i, one_batch_data in enumerate(train_loader):
        out = model(one_batch_data)
        print(out.shape)
        break


image_text_data_list, image_text_data_dict = load_image_text_data()
tweet_text_data_list, tweet_text_data_dict = load_tweet_text_data()


class TextPreprocessor:
    def __init__(self, sentences, sen_len=30, w2v_path='../save/w2v.model'):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.idx2word)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self):
        print('--------------Start processing word embedding--------------')
        for i, word in enumerate(self.embedding.wv.index_to_key):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding.wv[word])

        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding("")
        self.add_embedding("")
        print("total words: {}".format(len(self.embedding_matrix)))
        print('--------------Done processing word embedding--------------')
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度
        if len(sentence) > self.sen_len:  # 多的直接截断
            sentence = sentence[:self.sen_len]
        else:  # 少的添加""
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx[""])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 把句子里面的字变成相对应的index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx[""])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def get_sentence_word2idx_dict(self, text_dict):
        sentence_dict = {}
        for tweet_id, sen in text_dict.items():
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx[""])

            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_dict[tweet_id] = torch.LongTensor(sentence_idx)

        return sentence_dict

    def get_sen_len(self):
        return self.sen_len

text_preprocessor = TextPreprocessor(tweet_text_data_list+image_text_data_list)
embedding_matrix = text_preprocessor.make_embedding()

if __name__ == '__main__':
    # sentence_embedding_dict = text_preprocessor.get_sentence_word2idx_dict(tweet_text_data_dict)
    # print(sentence_embedding_dict['1116043441652236288'].shape)
    load_image_text_data()
