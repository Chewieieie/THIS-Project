import json
import os
import shutil


def make_dirs():
    base_dir = '../dataset/train/'
    for i in range(6):
        os.mkdir(base_dir + str(i))

def make_test_dirs():
    base_dir = '../dataset/test/'
    for i in range(6):
        os.mkdir(base_dir + str(i))


def statistic_data_length():
    base_dir = '../dataset/train/'
    for i in range(6):
        path = base_dir + str(i)
        print('label {} has {} images'.format(i, len(os.listdir(path))))


def get_final_annotation(labels):
    temp = dict()
    for i in range(6):
        temp[i] = 0

    for item in labels:
        temp[item] += 1

    key_set = sorted(temp.keys(), key=lambda x: temp[x], reverse=True)
    return key_set[0]


def dispatch_images():
    img_base_dir = '../MMHS150K/img_resized/'
    img_dst_dir = '../dataset/train/'
    d = {}
    with open('../MMHS150K/MMHS150K_GT.json') as file:
        d = json.loads(file.readline())

    # 遍历字典
    for tweet_id, val_dict in d.items():
        labels = val_dict['labels']
        which_dir = get_final_annotation(labels)

        src_path = img_base_dir + str(tweet_id) + '.jpg'
        dst_path = img_dst_dir + str(which_dir) + '/' + str(tweet_id) + '.jpg'

        shutil.copyfile(src_path, dst_path)
        print('src_path: {} -------> dst_path: {}, final label is: {}'.format(src_path, dst_path, which_dir))


def build_test_dataset():

    for i in range(6):

        train_base_dir = '../dataset/train/' + str(i) + '/'
        test_base_dir = '../dataset/test/' + str(i) + '/'

        train_file_list = os.listdir(train_base_dir)
        test_length = int(len(train_file_list) * 0.1)
        print(test_length)
        for j in range(test_length):
            file_name = train_file_list[j]

            src_path = train_base_dir + file_name
            dst_path = test_base_dir + file_name

            shutil.move(src_path, dst_path)
            break

        break


if __name__ == '__main__':
    # make_test_dirs()
    build_test_dataset()
