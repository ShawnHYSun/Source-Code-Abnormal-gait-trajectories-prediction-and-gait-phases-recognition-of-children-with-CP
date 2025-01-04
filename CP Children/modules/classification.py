import numpy as np
from keras.models import Model
from keras.optimizers import Adam
from modules.feature_extraction import build_feature_extractor
from keras.layers import *
from keras import backend as K
from modules.data_processing import process_data
import config

X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp, normalize_train_data, normalize_test_data, labels_one_hot = process_data(
    config.input_file, config.output_file, config.target_count, config.amount_of_features)


def weighted_categorical_crossentropy(weights):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_true = K.clip(y_true, epsilon, 1. - epsilon)
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weighted_loss = K.sum(weights * cross_entropy, axis=-1)
        return weighted_loss
    return loss


def build_classifier():
    outputs = Dense(len(labels_one_hot.categories_[0]), activation='softmax')(build_feature_extractor())

    # Build and compile model
    ensemble_model = Model(inputs=[Input(shape=(config.window, config.amount_of_features)),
                                   Input(shape=(X_train_mlp.shape[1],))],
                           outputs=outputs)

    class_weights = np.sum(normalize_train_data, axis=0) / float(len(normalize_train_data))
    class_weights = 1.0 / class_weights
    class_weights = class_weights / np.sum(class_weights)

    print(len(labels_one_hot.categories_[0]))
    print("Class Weights:", class_weights)

    ensemble_model.compile(loss=weighted_categorical_crossentropy(class_weights), optimizer=Adam(),
                           metrics=['accuracy'])

    # Train the model
    history = ensemble_model.fit([X_train_cnn, X_train_mlp], normalize_train_data,
                                 epochs=config.epoch, batch_size=config.batch_size,
                                 validation_data=([X_test_cnn, X_test_mlp], labels_one_hot), shuffle=False)

    # Evaluate the model
    results = ensemble_model.evaluate([X_test_cnn, X_test_mlp], labels_one_hot)
    print(f'Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}')

    # Predict on the test set
    y_test_pred = ensemble_model.predict([X_test_cnn, X_test_mlp])
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    y_test_true_labels = np.argmax(labels_one_hot, axis=1)

    return y_test_pred_labels, y_test_true_labels, history, ensemble_model