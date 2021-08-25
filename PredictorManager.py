import torch
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

    def train(self, train_db=None, file_path="", label_name='Open', basic_transform=None):
        self.train_db = VPDataset(file_path, label_name=label_name, basic_transform=basic_transform) if train_db is None else train_db
        self.value_predictor = ValuePredictor(self.train_db.attributes_len)
        print("Device set to:", self.value_predictor.device)
        self.value_predictor = self.value_predictor.to(self.value_predictor.device)

        loss_function = nn.MSELoss()
        optimizer = optim.Adam(self.value_predictor.parameters(), lr=Constants.LEARNING_RATE)

        train_dataloader = DataLoader(self.train_db, Constants.BATCH_SIZE, True)

        print("Training Started")
        batch_number = 0
        full_loss_list = []
        epoch_loss_list = []
        for epoch in range(Constants.EPOCHS):
            for i, input_data in enumerate(train_dataloader):
                histories, true_labels = input_data
                predicted_labels = self.value_predictor(histories)

                loss = loss_function(predicted_labels, true_labels.to(self.value_predictor.device, dtype=torch.float))
                full_loss_list.append(loss.item())
                batch_number += 1
                loss.backward()

                optimizer.step()
                self.value_predictor.zero_grad()

                if i == 0:
                    t = true_labels
                    p = predicted_labels
                    l = loss.item()

            # plt.scatter(t.cpu().detach().numpy() / 1.57861029e-05, p.cpu().detach().numpy() / 1.57861029e-05)
            # plt.title("Epoch {}".format(epoch))
            # plt.plot([0, 60000], [0, 60000], color="red")
            # plt.savefig("Plots/Epoch{}.png".format(epoch))
            # plt.cla()

            # Full Loss List
            # plt.plot(list(range(batch_number + 1 - i, batch_number + 1)), full_loss_list[-i:])
            # plt.title("Epoch {}".format(epoch))
            # plt.savefig("Plots/Epoch{} FullLoss.png".format(epoch))

            # Epoch Loss List
            # epoch_loss_list.append(sum(full_loss_list[-i:])/i)
            # plt.plot(list(range(len(epoch_loss_list))),epoch_loss_list)
            # plt.title("Epoch {}".format(epoch))
            # plt.savefig("Plots/Epoch{} EpochLoss.png".format(epoch))
            # plt.cla()

            print('true:', str(t / 1.57861029e-05))
            print('predicted:', str(p / 1.57861029e-05))
            print('loss:', str(l))
            print_str = "Epoch {} Completed".format(epoch + 1)
            print(print_str)

        #############################################
        # torch.save(self.dp, "PreTrained_Models/dpAdvModel.pt")
        # with open("PreTrained_Models/dbAdvModel.pkl", 'wb') as f:
        #     pickle.dump(self.db, f)
        # print("Saved!")
        #############################################

        print("Train is done.")

    def test(self, test_db=None, file_path="", label_name='Open', basic_transform=None):
        test_db = VPDataset(file_path, label_name=label_name, basic_transform=basic_transform) if test_db is None else test_db
        test_dataloader = DataLoader(test_db, len(test_db.labeled_data), True)
        # next(test_dataloader)
