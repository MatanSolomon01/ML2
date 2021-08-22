import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import Constants
from VPDataset import VPDataset
from ValuePredictor import ValuePredictor


class PredictorManager:
    def __init__(self):
        self.train_db = None
        self.value_predictor = None

    def train(self, file_path, label_name='Open', **kwargs):
        self.train_db = VPDataset(file_path, label_name=label_name, **kwargs)
        self.value_predictor = ValuePredictor()
        print("Device set to:", self.value_predictor.device)
        self.value_predictor = self.value_predictor.to(self.value_predictor.device)

        # todo: start training

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.value_predictor.parameters(), lr=Constants.LEARNING_RATE)

        train_dataloader = DataLoader(self.train_db, Constants.BATCH_SIZE, True)

        # Training start
        print("Training Started")

        for epoch in range(Constants.EPOCHS):
            for i, input_data in enumerate(train_dataloader):
                histories, true_labels = input_data

                # words_tags = words_tags.squeeze(0)

                predicted_labels = self.value_predictor(histories)
                loss = loss_function(predicted_labels, true_labels)
                loss.backward()

                optimizer.step()
                self.value_predictor.zero_grad()

                # printable_loss += loss.item()
                if i == 0:
                    t = true_labels
                    p = predicted_labels
                    l = loss.item()

            print('true:', str(t))
            print('predicted:', str(p))
            print('loss:', str(l))
            print_str = "Epoch {} Completed".format(epoch + 1)
            print(print_str)

        #############################################
        # torch.save(self.dp, "PreTrained_Models/dpAdvModel.pt")
        # with open("PreTrained_Models/dbAdvModel.pkl", 'wb') as f:
        #     pickle.dump(self.db, f)
        # print("Saved!")
        #############################################
