import os
import torch
from PIL import Image
from commons.image_transforms import *


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        self.args = args
        self.filenames = get_file_names(self.args.datapath)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = torchvision.transforms.Resize((224, 224))(
            Image.open(os.path.join(self.args.datapath, 'train', self.filenames[idx])).convert('RGB')
        )
        return {
            'image1': tensorify(
                augmented_image(img)
            ),
            'image2': tensorify(
                augmented_image(img)
            )
        }


def get_file_names(data_path):
    with open(os.path.join(data_path, "train", "names.txt")) as f:
        return f.read().split('\n')
