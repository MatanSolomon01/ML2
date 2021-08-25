import random

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

import Constants
from Utils import Utils as U


class VPDataset(Dataset):
    def __init__(self, file_path=None, label_name='Open', basic_transform=None, labeled_data=None, attributes_len=None):
        if file_path is not None:
            self.labeled_data = []  # [(t([][][][][]), float),(),()]
            self.attributes_len = 0
            self.__load_data(file_path, label_name=label_name, basic_transform=basic_transform)
        elif labeled_data is not None:
            self.labeled_data = labeled_data  # [(t([][][][][]), float),(),()]
            self.attributes_len = attributes_len
        else:
            print("Constructor failed :(")

    def __load_data(self, file_path, label_name='Open', basic_transform=None):
        """

        :param file_path:
        :param kwargs:
        :return:
        """
        raw_data = pd.read_csv(file_path, header=0, date_parser=True)
        raw_data = U.basic_transform(raw_data, basic_transform)
        self.attributes_len = raw_data.shape[1]

        items_len = raw_data.shape[0]
        for i in range(Constants.WINDOWS_SIZE, items_len):
            data = torch.tensor(raw_data[i - Constants.WINDOWS_SIZE:i].values, dtype=torch.float32)
            label = raw_data.loc[i, label_name]
            self.labeled_data.append(tuple([data, label]))

    def __len__(self):
        return len(self.labeled_data)

    def __getitem__(self, index):
        return self.labeled_data[index]

    def split_train_test(self, train_volume, method="CHRONOLOGICAL"):
        """
        starts from train
        :param train_volume:
        :return:
        """
        total_len = len(self.labeled_data)
        train_len = round(total_len * train_volume)
        # test_len = total_len - train_len

        if method == "CHRONOLOGICAL":
            train = VPDataset(labeled_data=self.labeled_data[:train_len], attributes_len=self.attributes_len)
            test = VPDataset(labeled_data=self.labeled_data[train_len:], attributes_len=self.attributes_len)
            return train, test

        elif method == "RANDOM":
            shuffled = random.sample(self.labeled_data, total_len)
            train = shuffled[:train_len]
            test = shuffled[train_len:]
            return train, test

        else:
            print("Invalid method, Split failed. Please use method from Constants.split_methods")
