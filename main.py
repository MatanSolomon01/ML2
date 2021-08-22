import Constants
from PredictorManager import PredictorManager
from Utils import Utils as U


def main():
    U.print_constants()
    manager = PredictorManager()
    basic_transform = {'columns_index_to_drop': [0, 5], 'remove_null_rows': True}
    Constants.LSTM_INPUT_SIZE = Constants.LSTM_INPUT_SIZE - len(basic_transform['columns_index_to_drop'])
    manager.train(file_path='Data/BTC-USD.csv', label_name='Open', **basic_transform)


if __name__ == '__main__':
    main()
