import random

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

import Constants
from Utils import Utils as U
from Utils import DfOperations as DfOp


class VPDataset(Dataset):
    def __init__(self, file_path=None, label_name=None, label_index='', predicted_len=Constants.LINEAR_LAYER,
                 basic_transform=None, labeled_data=None, attributes_len=None):
        if label_name is None:
            label_name = ['Open', 'High', 'Low', 'Close', 'Volume']
        if file_path is not None:
            self.labeled_data = []  # [(t([][][][][]), float),(t([][][][][]), float),...,(t([][][][][]), float)]
            self.attributes_len = 0
            self.__load_data(file_path, label_name=label_name, basic_transform=basic_transform)
        elif labeled_data is not None:
            self.labeled_data = labeled_data  # [(t([][][][][]), float),(),()]
            self.attributes_len = attributes_len
            self.label_index = label_index
            self.predicted_len = predicted_len
        else:
            print("Constructor failed :(")

    def __load_data(self, file_path, label_name, basic_transform=None):
        """
        :param file_path:
        :param kwargs:
        :return:
        """
        if label_name is None:
            label_name = ['Open', 'High', 'Low', 'Close', 'Volume']
        print(file_path)  # todo: remove
        raw_data = pd.read_csv(file_path, header=0, date_parser=True)
        raw_data, returned = U.basic_transform(raw_data, basic_transform)
        dates = raw_data['Date']
        raw_data = raw_data.drop('Date', axis=1)
        self.predicted_len = len(label_name)
        self.label_index = [raw_data.columns.get_loc(label) for label in label_name]
        items_len = raw_data.shape[0]
        new_df = ((raw_data[1:].reset_index(drop=True) - raw_data[:-1]) / raw_data[:-1]) \
            .rename(columns={col: col + '_diff' for col in raw_data.columns})
        raw_data = raw_data[1:].reset_index(drop=True)
        raw_data = pd.concat([raw_data, new_df], axis=1)
        self.attributes_len = raw_data.shape[1]

        for i in range(Constants.WINDOWS_SIZE + Constants.PREDICTION_SIZE, items_len):
            data = torch.tensor(
                raw_data[i - Constants.WINDOWS_SIZE - Constants.PREDICTION_SIZE:i - Constants.PREDICTION_SIZE].values,
                dtype=torch.float32)
            labels = torch.tensor(raw_data.loc[i - Constants.PREDICTION_SIZE:i - 1, label_name].values,
                                  dtype=torch.float32)
            self.labeled_data.append([data, labels, list(dates[i - Constants.PREDICTION_SIZE:i].values),
                                      list(dates[
                                           i - Constants.WINDOWS_SIZE - Constants.PREDICTION_SIZE:i - Constants.PREDICTION_SIZE].values)])

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
            train = VPDataset(labeled_data=self.labeled_data[:train_len], attributes_len=self.attributes_len,
                              label_index=self.label_index, predicted_len=self.predicted_len)
            test = VPDataset(labeled_data=self.labeled_data[train_len:], attributes_len=self.attributes_len,
                             label_index=self.label_index, predicted_len=self.predicted_len)
            return train, test

        elif method == "RANDOM":
            shuffled = random.sample(self.labeled_data, total_len)
            train = VPDataset(labeled_data=shuffled[:train_len], attributes_len=self.attributes_len,
                              label_index=self.label_index, predicted_len=self.predicted_len)
            test = VPDataset(labeled_data=shuffled[train_len:], attributes_len=self.attributes_len,
                             label_index=self.label_index, predicted_len=self.predicted_len)
            return train, test

        else:
            print("Invalid method, Split failed. Please use method from Constants.split_methods")

    def normalize(self):
        min_ele = torch.min(self.labeled_data[0][0], 0).values
        max_ele = torch.max(self.labeled_data[0][0], 0).values
        for i in range(1, len(self.labeled_data)):
            min_ele = torch.minimum(torch.min(self.labeled_data[i][0], 0).values, min_ele)
            max_ele = torch.maximum(torch.max(self.labeled_data[i][0], 0).values, max_ele)
        scale = 1 / (max_ele - min_ele)
        min_ele_temp = min_ele[self.label_index]
        scale_temp = scale[self.label_index]
        for i in range(0, len(self.labeled_data)):
            self.labeled_data[i][0] = (self.labeled_data[i][0] - min_ele) * scale
            self.labeled_data[i][1] = (self.labeled_data[i][1] - min_ele_temp) * scale_temp
            # self.labeled_data[i][0] = (-1+2*(self.labeled_data[i][0]-min_ele)*scale)
            # self.labeled_data[i][1] = -1+2*(self.labeled_data[i][1]-min_ele_temp)*scale_temp
        self.normalized = {"Shift": min_ele, "Scale": scale}

    def fit_norm(self, args_dict):
        min_ele = args_dict["Shift"]
        scale = args_dict["Scale"]
        min_ele_temp = min_ele[self.label_index]
        scale_temp = scale[self.label_index]
        for i in range(0, len(self.labeled_data)):
            self.labeled_data[i] = ((self.labeled_data[i][0] - min_ele) * scale,
                                    (self.labeled_data[i][1] - min_ele_temp) * scale_temp,
                                    self.labeled_data[i][2],
                                    self.labeled_data[i][3])
