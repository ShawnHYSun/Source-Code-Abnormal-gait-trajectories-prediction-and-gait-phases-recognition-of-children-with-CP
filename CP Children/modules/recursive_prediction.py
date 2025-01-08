import numpy as np
from keras.models import Model
from sklearn.metrics import r2_score
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras import backend as K
from modules.feature_extraction import build_feature_extractor
from keras.layers import *
from modules.data_processing import process_data
import config

X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp, normalize_train_data, normalize_test_data = process_data(
    config.input_file, config.output_file, config.target_count, config.amount_of_features)


# Custom callback to calculate R2 for train and validation sets during training
class R2ScoreCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions for train and validation sets
        y_train_predict = self.model.predict([X_train_cnn, X_train_mlp])
        y_val_predict = self.model.predict([X_test_cnn, X_test_mlp])

        # Calculate R2 for train and validation sets
        train_r2 = r2_score(normalize_train_data, y_train_predict)
        val_r2 = r2_score(normalize_test_data, y_val_predict)

        # Log the R2 scores
        logs['train_r2'] = train_r2
        logs['val_r2'] = val_r2


# Train the model with the custom callback
r2_callback = R2ScoreCallback()


# Recursive Prediction Function
def recursive_predict_with_ekf(model, X_input, n_steps, ekf_params):
    """

    Recursive prediction with Extended Kalman Filter (EKF) correction.

    Parameters:
    model (keras.Model): The trained model.
    X_input (numpy array): The initial input data (shape: [1, window, features]).
    n_steps (int): The number of time steps to predict.
    ekf_params (dict): EKF parameters including:
        - 'F': State transition matrix (function or array).
        - 'H': Observation matrix (function or array).
        - 'Q': Process noise covariance.
        - 'R': Measurement noise covariance.
        - 'P': Initial error covariance.

    Returns:
    numpy array: The corrected predicted values for the next `n_steps` time steps.


    """
    predictions = []
    input_seq_cnn = X_input.copy()
    input_seq_mlp = X_input.reshape(1, -1)  # Flatten for MLP input

    # Initialize EKF variables
    F = ekf_params['F']  # State transition matrix or function
    H = ekf_params['H']  # Observation matrix or function
    Q = ekf_params['Q']  # Process noise covariance
    R = ekf_params['R']  # Measurement noise covariance
    P = ekf_params['P']  # Initial error covariance
    x_est = X_input[0, -1, 0]  # Initial state estimate (last value in input sequence)

    # Recursive prediction with EKF
    for _ in range(n_steps):
        # Model Prediction
        pred = model.predict([input_seq_cnn, input_seq_mlp])[0, 0]

        # EKF Prediction
        if callable(F):
            x_pred = F(x_est)  # Non-linear state transition
        else:
            x_pred = np.dot(F, x_est)  # Linear state transition

        P_pred = np.dot(F, np.dot(P, F.T)) + Q

        # EKF Update
        z = pred  # The model's raw prediction serves as the observation
        if callable(H):
            y = z - H(x_pred)  # Innovation
            H_jacobian = H(x_pred, jacobian=True)  # If H is non-linear, get its Jacobian
        else:
            y = z - np.dot(H, x_pred)  # Innovation
            H_jacobian = H

        S = np.dot(H_jacobian, np.dot(P_pred, H_jacobian.T)) + R
        K = np.dot(P_pred, np.dot(H_jacobian.T, np.linalg.inv(S)))  # Kalman gain
        x_est = x_pred + np.dot(K, y)  # Corrected state estimate
        P = P_pred - np.dot(K, np.dot(H_jacobian, P_pred))  # Update error covariance

        # Append corrected prediction
        predictions.append(x_est)

        # Update input sequence for next prediction
        input_seq_cnn = np.roll(input_seq_cnn, shift=-1, axis=1)
        input_seq_cnn[0, -1, 0] = x_est  # Update with EKF-corrected prediction
        input_seq_mlp = input_seq_cnn.reshape(1, -1)  # Flatten for MLP input

    return np.array(predictions)

def F(x):
    return x + 0.15 * np.sin(x)

def H(x, jacobian=False):
    if jacobian:
        # Return the Jacobian of H with respect to x
        return np.array([[1]])
    return x  # Observation model

# Define EKF parameters
ekf_params = {
    'F': np.array([[1]]),  # State transition matrix
    'H': np.array([[1]]),  # Observation matrix
    'Q': np.array([[0.05]]),  # Process noise covariance
    'R': np.array([[0.125]]),  # Measurement noise covariance
    'P': np.array([[1]]),  # Initial error covariance
}


def build_recursive_predictor():
    # Sum along the axis to get the weighted sum
    attention = Lambda(lambda x: K.sum(x, axis=-2), name='attention')(build_feature_extractor())

    # Final output layer
    outputs = Dense(1, activation='tanh')(attention)

    # Build and compile the model
    ensemble_model = Model(inputs=[Input(shape=(config.window, config.amount_of_features)),
                                   Input(shape=(X_train_mlp.shape[1],))],
                           outputs=outputs)
    ensemble_model.compile(loss=config.loss_prediction, optimizer=Adam(learning_rate=config.learning_rate),
                           metrics=['accuracy'])

    # Show the structure of the model
    ensemble_model.summary()

    # Train the model
    history = ensemble_model.fit([X_train_cnn, X_train_mlp], normalize_train_data, epochs=config.epoch, 
                                 batch_size=config.batch_size, shuffle=False,
                                 validation_data=([X_test_cnn, X_test_mlp], normalize_test_data),
                                 callbacks=[r2_callback])

    # Calculate test accuracy and test loss manually at the end of training
    test_loss, test_accuracy = ensemble_model.evaluate([X_test_cnn, X_test_mlp], normalize_test_data)

    y_test_predict = ensemble_model.predict([X_test_cnn, X_test_mlp])

    n_steps = 10  # Number of steps to predict

    # Start with the last data point from the test set for recursive prediction
    X_input = X_test_cnn[-1].reshape(1, config.window, config.amount_of_features)  # Get the last sequence

    # Perform recursive prediction
    predictions_recursive_ekf = recursive_predict_with_ekf(ensemble_model, X_input, n_steps, ekf_params)

    # Flatten the predictions to make them 1D
    predictions_recursive_ekf = predictions_recursive_ekf.flatten()

    return predictions_recursive_ekf, history
