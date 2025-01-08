import matplotlib as plt
import numpy as np
import config
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from keras.callbacks import Callback
from modules.data_processing import process_data
from modules.recursive_prediction import build_recursive_predictor

X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp, normalize_train_data, normalize_test_data, labels_one_hot = \
    process_data(config.input_file, config.output_file, config.target_count, config.amount_of_features)

predictions_recursive_ekf, history= build_recursive_predictor()


# Custom callback to calculate R2 for train and validation sets during training
class R2ScoreCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions for train and validation sets
        y_train_predict = self.model.predict([X_train_cnn, X_train_mlp])
        y_val_predict = self.model.predict([X_test_cnn, X_test_mlp])

        # Calculate R2 for train and validation sets
        train_r2 = r2_score(normalize_train_data, y_train_predict)
        val_r2 = r2_score(normalize_train_data, y_val_predict)

        # Log the R2 scores
        logs['train_r2'] = train_r2
        logs['val_r2'] = val_r2


def plot_results_prediction():
    # Train the model with the custom callback
    r2_callback = R2ScoreCallback()

    plt.rcParams['font.family'] = 'Times New Roman'

    # Plot the predictions
    plt.figure(figsize=(15, 6))
    blue_color = (52 / 255, 120 / 255, 185 / 255)
    orange_color = (255 / 255, 145 / 255, 55 / 255)
    plt.plot(np.arange(len(normalize_test_data)), normalize_test_data, label='True', color=blue_color)
    plt.plot(np.arange(len(normalize_test_data), len(normalize_test_data) + 10), predictions_recursive_ekf,
             linestyle='dashed', color=orange_color)
    plt.title('Actual vs Predicted Data')
    plt.xlim(0)
    plt.xlabel('Time Steps')
    plt.ylabel('Joint Angles')
    plt.legend()
    plt.show()

    # Plot training and validation loss and accuracy
    plt.figure(figsize=(12, 6))
    # Plot training loss (custom green dotted line)
    plt.plot(history.history['loss'], color='#00C700', linestyle='--', label='Training Loss', linewidth=2)
    # Plot training accuracy (custom red dotted line)
    plt.plot(history.history['accuracy'], color='#FE007D', linestyle='--', label='Training R2', linewidth=2)
    # Plot validation loss (custom solid red line)
    plt.plot(history.history['val_loss'], color='#FE007D', linestyle='-', label='Validation Loss', linewidth=2)
    # Plot validation accuracy (custom solid green line)
    plt.plot(history.history['val_accuracy'], color='#00C700', linestyle='-', label='Validation R2', linewidth=2)

    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    # plt.title('Training and Validation Loss and Accuracy')
    plt.legend()
    plt.show()

    # Calculate MAE
    test_mae = mean_absolute_error(normalize_test_data, predictions_recursive_ekf)
    r2 = r2_score(normalize_test_data, predictions_recursive_ekf)

    # Extract loss and R2 from the history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_r2 = history.history['train_r2']
    val_r2 = history.history['val_r2']

    # Plot Loss and R2 for train and validation sets in a single plot
    plt.figure(figsize=(10, 6))

    # Create the first axis for Loss
    ax1 = plt.gca()
    ax1.plot(train_loss, label='Train Loss', color='#00C700', linestyle='--', linewidth=1.5)
    ax1.plot(val_loss, label='Validation Loss', color='#FE007D', linestyle='-', linewidth=1.5)
    ax1.set_xlabel('Epochs', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Prediction Loss', color='black', fontsize=16, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second axis for R2 that shares the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(train_r2, label='Train R2', color='#FE007D', linestyle='--', linewidth=1.5)
    ax2.plot(val_r2, label='Validation R2', color='#00C700', linestyle='-', linewidth=1.5)
    ax2.set_ylabel('R2', color='black', fontsize=16, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='black')

    # Add titles and legends
    plt.title('Loss and R2 vs Epochs')
    # Combine the legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    plt.legend()
    plt.show()

    # Print the results
    print(f'Test MAE: {test_mae:.2f}')
    print(f'R2 Score:", {r2:.2f}')
