import json
import os.path

import matplotlib.pyplot as plt


def unit_test():
    with open('../MMHS150K/MMHS150K_GT.json') as file:
        d = dict()
        d = json.loads(file.readline())

        print(d)

        tweet_id = '1024825773641879554'
        print(d[tweet_id])

        show_one_image(tweet_id)


def show_one_image(id):
    img_path = '../MMHS150K/img_resized/' + str(id) + '.jpg'
    print('img_path: {}'.format(img_path))

    img = plt.imread(img_path)
    plt.figure('example')
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    unit_test()
