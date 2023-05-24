import os
import json
import torch
from PIL import Image

from commons.image_transforms import *


class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        self.args = args

        with open(os.path.join(args.datapath, 'train', 'train.json')) as f:
            self.filedict = json.load(f)

        with open(os.path.join(args.datapath, 'mapper.json')) as f:
            self.mapper = json.load(f)

        self.filenames = list(self.filedict)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return {
            'image': tensorify(
                torchvision.transforms.Resize((224, 224))(
                    Image.open(os.path.join(self.args.datapath, 'train', self.filenames[idx])).convert('RGB')
                )
            ),
            'label': self.mapper[self.filedict[self.filenames[idx]]]
        }


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, args):
        self.args = args

        with open(os.path.join(args.datapath, 'test', 'test.json')) as f:
            self.filedict = json.load(f)

        with open(os.path.join(args.datapath, 'mapper.json')) as f:
            self.mapper = json.load(f)

        self.filenames = list(self.filedict)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return {
            'image': tensorify(
                torchvision.transforms.Resize((224, 224))(
                    Image.open(os.path.join(self.args.datapath, 'test', self.filenames[idx])).convert('RGB')
                )
            ),
            'label': self.mapper[self.filedict[self.filenames[idx]]]
        }