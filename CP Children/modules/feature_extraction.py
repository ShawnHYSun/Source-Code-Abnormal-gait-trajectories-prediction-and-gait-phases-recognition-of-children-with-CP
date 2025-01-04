from keras.layers import *
from keras import backend as K
from modules.data_processing import process_data
import config

X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp, normalize_train_data, normalize_test_data = process_data(
    config.input_file, config.output_file, config.target_count, config.amount_of_features)

def cnn_lstm_branch():
    # CNN-BiLSTM branch
    inputs_cnn = Input(shape=(config.window, config.amount_of_features))
    cnn_lstm = Conv1D(filters=config.lstm_units, kernel_size=1, activation='relu')(inputs_cnn)
    cnn_lstm = MaxPooling1D(pool_size=config.window)(cnn_lstm)
    cnn_lstm = Dropout(config.dropout)(cnn_lstm)
    cnn_lstm = Bidirectional(LSTM(config.lstm_units, activation='sigmoid'), name='bilstm')(cnn_lstm)
    return cnn_lstm

def mlp_branch():
    # MLP branch
    inputs_mlp = Input(shape=(X_train_mlp.shape[1],))
    mlp = Dense(config.mlp_units * 2, activation='relu')(inputs_mlp)
    mlp = Dropout(config.dropout)(mlp)
    mlp = Dense(config.mlp_units, activation='relu')(mlp)
    return mlp

def build_feature_extractor():
    # Concatenate both branches
    merged = concatenate([cnn_lstm_branch(), mlp_branch()])

    # Attention layer
    attention = Dense(1, activation='sigmoid', name='attention_vec')(merged)
    attention = Reshape((-1, 1))(attention)
    attention = GlobalAveragePooling1D()(attention)
    attention = RepeatVector(merged.shape[1])(attention)
    attention = Permute([2, 1])(attention)

    # Reshape attention tensor for element-wise multiplication
    attention = Reshape((merged.shape[1], 1))(attention)

    # Apply attention weights directly to the merged output
    attention_output = Multiply()([merged, attention])

    return attention_output
