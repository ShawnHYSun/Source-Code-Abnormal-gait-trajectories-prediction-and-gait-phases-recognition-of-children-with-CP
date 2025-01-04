import matplotlib as plt
import numpy as np
import config
import seaborn as sns
from modules.data_processing import process_data
from sklearn.metrics import classification_report, confusion_matrix
from modules.classification import build_classifier
from sklearn.preprocessing import LabelEncoder


X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp, normalize_train_data, normalize_test_data, labels_one_hot = \
    process_data(config.input_file, config.output_file, config.target_count, config.amount_of_features)

y_test_pred_labels, y_test_true_labels, history, ensemble_model = build_classifier()

label_encoder = LabelEncoder()
y_train_labels = label_encoder.fit_transform(labels_one_hot)
y_test_labels = label_encoder.transform(y_test_true_labels)

def draw_confusion_matrix(cm, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=300):
    """
    Draw a confusion matrix and compute metrics like accuracy, precision, recall, and F1 score.

    @param cm: The precomputed confusion matrix (e.g., normalized confusion matrix).
    @param label_name: List of label names (e.g., ['cat', 'dog', 'flower', ...]).
    @param title: Title of the plot.
    @param pdf_save_path: Path to save the plot (e.g., 'plot.png', 'plot.pdf').
    @param dpi: Resolution of the saved image.
    """
    # Set global font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # Calculate Accuracy, Precision, Recall, and F1 Score
    accuracy = np.trace(cm) / np.sum(cm)
    precisions = np.diag(cm) / np.sum(cm, axis=0)
    recalls = np.diag(cm) / np.sum(cm, axis=1)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    macro_f1 = np.mean(f1_scores)

    # Print the computed metrics for clarity
    print(f"Precisions: {precisions}")
    print(f"Recalls: {recalls}")
    print(f"F1 Scores: {f1_scores}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    # Plot confusion matrix
    plt.imshow(cm, cmap='Blues')
    plt.xlabel("Predicted label", fontweight='bold', fontsize=14)
    plt.ylabel("True label", fontweight='bold', fontsize=14)
    plt.yticks(range(len(label_name)), label_name)
    plt.xticks(range(len(label_name)), label_name, rotation=45)
    plt.tight_layout()
    plt.colorbar()

    # Annotate the matrix with values
    for i in range(len(label_name)):
        for j in range(len(label_name)):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # White font for diagonal, black for others
            value = float(format('%.2f' % cm[j, i]))  # Convert to float and format
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # Save the plot
    if pdf_save_path:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


def plot_results_classification():
    # Evaluate the model
    results = ensemble_model.evaluate([X_test_cnn, X_test_mlp], normalize_test_data)
    print(f'Test Loss: {results[0]:.4f}, Test Accuracy: {results[1]:.4f}')

    # Predict on the test set
    y_test_pred = ensemble_model.predict([X_test_cnn, X_test_mlp])
    y_test_pred_labels = np.argmax(y_test_pred, axis=1)
    y_test_true_labels = np.argmax(normalize_test_data, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test_true_labels, y_test_pred_labels)

    # Label names
    labels = ['KF', 'KE', 'HE', 'HF', 'SF', 'SE']

    # Use the function to plot the confusion matrix and calculate the metrics
    draw_confusion_matrix(cm=cm, label_name=labels, title="Confusion Matrix",pdf_save_path=None, dpi=300)


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
    report = classification_report(y_test_true_labels, y_test_pred_labels, target_names=label_encoder.classes_)
    print("Classification Report:")
    print(report)