from collections import OrderedDict

from PredictorManager import PredictorManager
from Utils import DfOperations as DfOp
from Utils import Utils as U


def main():
    U.print_constants()
    manager = PredictorManager()

    basic_transform = OrderedDict([(DfOp.drop_columns, [[0, 5]]),
                                   (DfOp.drop_null_rows, []),
                                   (DfOp.normalize, ['MinMax']),
                                   (DfOp.omit_last, [60])])

    manager.train(file_path='Data/BTC-USD.csv', label_name='Open', basic_transform=basic_transform)


if __name__ == '__main__':
    main()
