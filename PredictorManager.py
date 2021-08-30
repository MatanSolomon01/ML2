import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from matplotlib.pyplot import figure
import numpy as np

import Constants
from VPDataset import VPDataset
from ValuePredictor import ValuePredictor


class PredictorManager:
    def __init__(self):
        self.train_db = None
        self.value_predictor = None

    def train(self, train_db=None, file_path="", label_name=None, basic_transform=None):
        if label_name is None:
            label_name = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.train_db = VPDataset(file_path, label_name=label_name, basic_transform=basic_transform) \
            if train_db is None else train_db
        self.value_predictor = ValuePredictor(self.train_db.attributes_len, self.train_db.predicted_len)
        print("Device set to:", self.value_predictor.device)
        self.value_predictor = self.value_predictor.to(self.value_predictor.device)

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.value_predictor.parameters(), lr=Constants.LEARNING_RATE)

        train_dataloader = DataLoader(self.train_db, Constants.BATCH_SIZE, True)
        scale = self.train_db.normalized["Scale"].to(self.value_predictor.device, dtype=torch.float)
        shift = self.train_db.normalized["Shift"].to(self.value_predictor.device, dtype=torch.float)
        print("Training Started")
        batch_number = 0
        full_loss_list = []
        # epoch_loss_list = []
        for epoch in range(Constants.EPOCHS):
            for i, input_data in enumerate(train_dataloader):
                histories, true_labels, _, _ = input_data
                histories = histories.to(self.value_predictor.device)
                true_labels = true_labels.to(self.value_predictor.device, dtype=torch.float)
                list_predicted = []
                for predicted_day_index in range(Constants.PREDICTION_SIZE):
                    predicted_labels = self.value_predictor(histories)
                    list_predicted.append(predicted_labels.unsqueeze(1))
                    histories = torch.cat([histories[:, 1:, :], predicted_labels.unsqueeze(1)], dim=1)
                tensor_predicted = torch.cat(list_predicted, dim=1)
                loss = loss_function(tensor_predicted, true_labels)
                full_loss_list.append(loss.item() * (scale[self.train_db.label_index]) ** 2)
                batch_number += 1
                loss.backward()

                optimizer.step()
                self.value_predictor.zero_grad()

            print_str = "Epoch {} Completed".format(epoch + 1)
            print(print_str)

        print("Train is done.")

    def norm_loss(self, history, true_labels, predicted_labels, scale, shift):
        history = history / scale[self.train_db.label_index] + shift[self.train_db.label_index]
        true_labels = true_labels / scale[self.train_db.label_index] + shift[self.train_db.label_index]
        predicted_labels = predicted_labels / scale[self.train_db.label_index] + shift[self.train_db.label_index]
        avg_loss = torch.mean(abs((predicted_labels - true_labels) / true_labels), axis=0)
        mse_loss = torch.mean(((predicted_labels - true_labels) ** 2), axis=0)
        return history, true_labels, predicted_labels, mse_loss, avg_loss

    def plot_pred_by_date(self, history, true_labels, predicted_labels, dates, history_date):
        names = ['a', 'b', 'c', 'd', 'e']
        indexes = list(history_date)
        indexes.extend(list(dates))
        history = history.cpu().detach()
        true_labels = true_labels.cpu().detach()
        predicted_labels = predicted_labels.cpu().detach()
        f, axes = plt.subplots(5, 1, figsize=(30, 20), constrained_layout=True)
        for i, ax in enumerate(axes):
            ax.plot(indexes, torch.cat([history[:, i], true_labels[:, i]]), label="True", marker='o')
            ax.plot(dates, predicted_labels[:, i], label="Predicted", marker='o')
            ax.grid(True)
            ax.set_xticks([indexes[i] for i in range(0, len(indexes), 5)])
            ax.set_xticklabels([indexes[i] for i in range(0, len(indexes), 5)], rotation=45, ha='right')
            ax.set_title(names[i])
            ax.legend()
            ax.set_xlabel("Dates")
            ax.set_ylabel("Values")

        f.suptitle('True vs Predicted values', fontsize=16)
        f.show()

    def test(self, test_db=None, file_path="", label_name='Open', basic_transform=None):
        scale = self.train_db.normalized["Scale"].to(self.value_predictor.device, dtype=torch.float)
        shift = self.train_db.normalized["Shift"].to(self.value_predictor.device, dtype=torch.float)
        test_db = VPDataset(file_path, label_name=label_name, basic_transform=basic_transform) \
            if test_db is None else test_db
        test_dataloader = DataLoader(test_db, len(test_db.labeled_data), False)
        histories, true_labels, prediction_dates, histories_dates = list(test_dataloader)[0]
        histories = histories.to(self.value_predictor.device)
        histories_temp = torch.clone(histories)
        prediction_dates = list(zip(*prediction_dates))
        histories_dates = list(zip(*histories_dates))
        true_labels = true_labels.to(self.value_predictor.device, dtype=torch.float)
        list_predicted = []
        for predicted_day_index in range(Constants.PREDICTION_SIZE):
            predicted_labels = self.value_predictor(histories_temp)
            list_predicted.append(predicted_labels.unsqueeze(1))
            histories_temp = torch.cat([histories_temp[:, 1:, :], predicted_labels.unsqueeze(1)], dim=1)
        tensor_predicted = torch.cat(list_predicted, dim=1)
        histories, true_labels, predicted_labels, mse_loss, avg_loss = self.norm_loss(histories, true_labels,
                                                                                      tensor_predicted, scale, shift)
        print('true:', str(true_labels[30]))
        print('predicted:', str(predicted_labels[30]))
        print('MSE loss', str(mse_loss))
        print('Average loss in percentage: ', str(avg_loss))
        for i in [3, 5, 10, 30, 50, 70]:
            self.plot_pred_by_date(histories[i], true_labels[i], predicted_labels[i], prediction_dates[i],
                                   histories_dates[i])
