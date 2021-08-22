import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

import Constants
from Utils import Utils as U


class VPDataset(Dataset):
    def __init__(self, file_path, label_name='Open', basic_transform=None):
        self.labeled_data = []  # [(t([][][][][]), float),(),()]
        self.attributes_len = 0
        self.__load_data(file_path, label_name=label_name, basic_transform=basic_transform)

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
