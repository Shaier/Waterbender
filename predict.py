import argparse
import os

from fastai.vision import open_image, Image

from model import get_model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    learn = get_model(args.input_dir)
    learn.load('2b')  # path to weights

    for pic in os.listdir(args.input_dir):
        img_name = pic.replace('.jpg', '')  # note that my images were jpg format
        #print('Processing image: {}'.format(img_name))
        img = open_image(pic)
        _, img_hr, b = learn.predict(img)

        # save image
        Image(img_hr).save(args.output_dir + '/' + img_name + '_prediction.jpg')
