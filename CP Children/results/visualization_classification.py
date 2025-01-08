import matplotlib.pyplot as plt
import numpy as np
import config
import seaborn as sns
from modules.data_processing import process_data
from sklearn.metrics import classification_report, confusion_matrix
from modules.classification import build_classifier
from sklearn.preprocessing import LabelEncoder

# Process the data
X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp, normalize_train_data, normalize_test_data, labels_one_hot = \
    process_data(config.input_file, config.output_file, config.target_count, config.amount_of_features)

# Build classifier
y_test_pred_labels, y_test_true_labels, history, ensemble_model = build_classifier()

# Decode one-hot labels to integers
y_test_true_labels = np.argmax(normalize_test_data, axis=1)  # True labels
y_test_pred_labels = np.argmax(y_test_pred_labels, axis=1)  # Predicted labels

def draw_confusion_matrix(cm, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """

    Draw a confusion matrix and compute metrics like accuracy, precision, recall, and F1 score.


    @param cm: The precomputed confusion matrix (e.g., normalized confusion matrix).
    @param label_name: List of label names (e.g., ['KF', 'KE', 'HE', 'HF', 'SF', 'SE']).
    @param title: Title of the plot.
    @param pdf_save_path: Path to save the plot (e.g., 'plot.png', 'plot.pdf').
    @param dpi: Resolution of the saved image.


    """
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Metrics
    accuracy = np.trace(cm) / np.sum(cm)
    precisions = np.diag(cm_normalized) / np.sum(cm_normalized, axis=0)
    recalls = np.diag(cm_normalized)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    macro_f1 = np.mean(f1_scores)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # Plot
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', xticklabels=label_name, yticklabels=label_name, fmt='.2f')
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.tight_layout()

    if pdf_save_path:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    plt.show()

def plot_results_classification():
    # Evaluate the model
    results = ensemble_model.evaluate([X_test_cnn, X_test_mlp], normalize_test_data)
    print(f'Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_test_true_labels, y_test_pred_labels)
    labels = ['KF', 'KE', 'HE', 'HF', 'SF', 'SE']
    draw_confusion_matrix(cm=cm, label_name=labels, title="Confusion Matrix")

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

    # Classification Report
    report = classification_report(y_test_true_labels, y_test_pred_labels, target_names=labels)
    print("Classification Report:")
    print(report)

# Call the function to visualize results
plot_results_classification()
