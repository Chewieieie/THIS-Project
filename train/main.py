from model.text_model import load_image_text_data, load_tweet_text_data, TextDataset, TextPreprocessor, LSTMNet, tweet_text_data_list
from torch.utils.data import DataLoader


def main():
    sen_len = 30
    train_x = tweet_text_data_list
    text_preprocess = TextPreprocessor(train_x, sen_len)
    embedding_matrix = text_preprocess.make_embedding()

    print('embedding_matrix: {}'.format(embedding_matrix.shape))

    train_x = text_preprocess.sentence_word2idx()

    train_dataset = TextDataset(data=train_x, label=None)

    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2)

    model = LSTMNet(embedding_matrix, 100)

    for i, one_batch_data in enumerate(train_loader):
        print(one_batch_data.shape)
        out = model(one_batch_data)
        print(out.shape)
        break


if __name__ == '__main__':
    main()
