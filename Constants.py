# ValuePredictor > init
LSTM_DEFAULT_INPUT_SIZE = 5
LSTM_HIDDEN_SIZE = 300
LSTM_NUM_LAYERS = 2
LINEAR_LAYER = 1

# Train
WINDOWS_SIZE = 60
LEARNING_RATE = 0.001
BATCH_SIZE = 20
EPOCHS = 50

# General
FILES_NAMES = ['Data/ADA-USD.csv', 'Data/BNB-USD.csv', 'Data/BTC-USD.csv', 'Data/ETH-USD.csv', 'Data/gold_values.csv']


# Split methods
class split_methods:
    SPLIT_CHRONOLOGICAL = "CHRONOLOGICAL"
    SPLIT_RANDOM = "RANDOM"
