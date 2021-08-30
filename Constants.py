# ValuePredictor > init
LSTM_DEFAULT_INPUT_SIZE = 5
LSTM_HIDDEN_SIZE = 450
LSTM_NUM_LAYERS = 3
LINEAR_LAYER = 5

# Train
WINDOWS_SIZE = 60
PREDICTION_SIZE = 4
LEARNING_RATE = 0.001
BATCH_SIZE = 40
EPOCHS = 60

# General
# FILES_NAMES = ['Data/ADA-USD.csv', 'Data/BNB-USD.csv', 'Data/BTC-USD.csv', 'Data/ETH-USD.csv', 'Data/gold_values.csv']
FILES_NAMES = ['Data/BTC-USD.csv']


# split methods
class split_methods:
    SPLIT_CHRONOLOGICAL = "CHRONOLOGICAL"
    SPLIT_RANDOM = "RANDOM"
