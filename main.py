from collections import OrderedDict

import Constants as C
from PredictorManager import PredictorManager
from Utils import DfOperations as DfOp
from Utils import Utils as U
from VPDataset import VPDataset
from scipy.stats import spearmanr


def main():
    U.print_constants()
    manager = PredictorManager()
    basic_transform = OrderedDict([(DfOp.drop_columns, [[5]]),
                                   (DfOp.drop_null_rows, [])])
    trains = []
    tests = []
    for file_name in C.FILES_NAMES:
        main_db = VPDataset(file_path=file_name, label_name=['Open', 'High', 'Low', 'Close', 'Volume'],
                            basic_transform=basic_transform)
        train_db, test_db = main_db.split_train_test(train_volume=0.8, method=C.split_methods.SPLIT_CHRONOLOGICAL)
        train_db.normalize()
        test_db.fit_norm(train_db.normalized)
        trains.append(train_db)
        tests.append(test_db)
        manager.train(train_db)
        manager.test(test_db=test_db)


if __name__ == '__main__':
    main()
