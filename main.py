from collections import OrderedDict

import Constants as C
from PredictorManager import PredictorManager
from Utils import DfOperations as DfOp
from Utils import Utils as U
from VPDataset import VPDataset


def main():
    U.print_constants()

    manager = PredictorManager()
    basic_transform = OrderedDict([(DfOp.drop_columns, [[0, 5]]),
                                   (DfOp.drop_null_rows, []),
                                   (DfOp.normalize, ['MinMax']),
                                   (DfOp.omit_last, [60])])

    for file_name in C.FILES_NAMES:
        main_db = VPDataset(file_path=file_name, label_name='Open', basic_transform=basic_transform)
        train_db, test_db = main_db.split_train_test(train_volume=0.7, method=C.split_methods.SPLIT_CHRONOLOGICAL)
        manager.train(train_db)
        manager.test(test_db=test_db)


if __name__ == '__main__':
    main()
