import argparse
import os
import random
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from train import train_dataset
from commons import image_transforms

parser = argparse.ArgumentParser(description="This is the command line interface for the linear evaluation model")

parser.add_argument('-datapath', type=str, help="Path to the data root folder which contains train and test folders")

args = parser.parse_args()


def show_images(num_images=15, rows=5, columns=3):
    dataset_path = os.path.join('..\\', args.datapath, 'test')
    image_list = os.listdir(dataset_path)

    image_list = random.sample(image_list, num_images)

    fig = plt.figure(figsize=(10, 10))

    for i, image_name in enumerate(image_list):
        image_path = os.path.join(dataset_path, image_name)
        image = Image.open(image_path)

        ax = fig.add_subplot(rows, columns, i + 1)

        ax.imshow(image)

    plt.show()


def show_processed_images(num_images=5, columns=2):
    args.datapath = os.path.join('..\\', args.datapath)
    ds = train_dataset.TrainDataset(args)
    image_pairs = []
    for index in [random.randint(0, len(ds)) for p in range(0, num_images)]:
        processed_image1 = image_transforms.deprocess_and_show(ds[index]['image1'])
        processed_image2 = image_transforms.deprocess_and_show(ds[index]['image2'])
        image_pairs.append((processed_image1, processed_image2))

    fig, axs = plt.subplots(len(image_pairs), columns, figsize=(12, 12))

    for i, pair in enumerate(image_pairs):
        image_1, image_2 = pair
        ax = axs[i, 0]
        ax.imshow(image_1)
        ax = axs[i, 1]
        ax.imshow(image_2)

    plt.show()

# show_images()
# show_processed_images()
