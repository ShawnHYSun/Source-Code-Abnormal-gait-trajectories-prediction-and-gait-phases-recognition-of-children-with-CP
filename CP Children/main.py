# haoyuansun@cuhk.edu.hk

# import the necessary libraries
from modules.feature_extraction import build_feature_extractor
from modules.recursive_prediction import build_recursive_predictor
from modules.classification import build_classifier
from modules.data_processing import process_data
from results.visualization_prediction import plot_results_prediction
from results.visualization_classification import plot_results_classification
import config

def main():
    # Feature extraction
    feature_extractor = build_feature_extractor()

    # Recursive prediction
    recursive_predictor = build_recursive_predictor(feature_extractor)

    # Classification
    classifier = build_classifier(feature_extractor)

    # Plot the results
    plot_results_prediction()
    plot_results_classification()


if __name__ == "__main__":
    main()
