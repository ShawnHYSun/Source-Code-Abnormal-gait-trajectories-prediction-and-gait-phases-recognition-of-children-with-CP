import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import random


# Function to apply SplineBalance
def apply_spline_balance(data, target_count):
    time_steps = data['time'].values
    angles = data['angle'].values
    sorted_indices = np.argsort(time_steps)
    time_steps_sorted = time_steps[sorted_indices]
    angles_sorted = angles[sorted_indices]

    _, unique_indices = np.unique(time_steps_sorted, return_index=True)
    time_steps_sorted_unique = time_steps_sorted[unique_indices]
    angles_sorted_unique = angles_sorted[unique_indices]

    spline = CubicSpline(time_steps_sorted_unique, angles_sorted_unique)

    new_time_steps = np.linspace(time_steps_sorted_unique.min(), time_steps_sorted_unique.max(), target_count)
    new_angles = spline(new_time_steps)

    augmented_data = pd.DataFrame({
        'time': new_time_steps,
        'angle': new_angles
    })

    return augmented_data


# Function to assign labels to gait phases (KF, KE, HE, HF, SF, SE)
def assign_labels_to_gait_phases(data):
    labels = []

    # Set your thresholds for knee and hip
    a_max1 = -15  # Example max flexion angle for knee (KF/KE)
    a_min = -75  # Example min flexion angle for knee
    b_max = 20  # Example threshold for hip angle

    # Iterate through the dataset to assign labels
    for _, row in data.iterrows():
        a = row['knee_angle']  # Adjust based on your column name
        b = row['hip_angle']  # Adjust based on your column name

        if a_max1 > a >= a_min and b > b_max:
            labels.append('KF')  # Knees start contact, hip angle greater than threshold
        elif a_max1 >= a >= a_min and b <= b_max:
            labels.append('KE')  # Knee moves from flexion to extension, hip angle lower than threshold
        elif b < 0 and a_max1 >= a >= a_min:
            labels.append('HE')  # Hip angle is lower than zero, knee reaches its minimum point
        elif b < 0 and a_max1 >= a > a_min:
            labels.append('HF')  # Hip angle is negative, knee still extending
        elif b >= 0 and a_max1 >= a >= a_min:
            labels.append('SF')  # Swing phase, knee extremum spotted
        elif a_max1 >= a >= a_min and b > b_max:
            labels.append('SE')  # Swing phase, knee continues extension and hip angle exceeds threshold
        else:
            labels.append('Unknown')  # If no condition matched, label as unknown (optional)

    data['label'] = labels
    return data


# Function to encode labels to one-hot encoding
def encode_labels_to_one_hot(data):
    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse=False)

    # Encode labels into numeric labels
    data['label_encoded'] = label_encoder.fit_transform(data['label'])

    # Apply one-hot encoding to the numeric labels
    labels_one_hot = one_hot_encoder.fit_transform(data['label_encoded'].values.reshape(-1, 1))

    return labels_one_hot


# Function to compute label proportions and scaling factors for each gait phase
def compute_label_proportions(data):
    label_counts = data['label'].value_counts()
    total_samples = len(data)
    label_proportions = label_counts / total_samples
    alpha = 1 / label_proportions.max()
    new_samples_count = int(alpha * total_samples)
    target_counts = label_proportions * new_samples_count

    return label_proportions, target_counts, alpha, new_samples_count


# Function for Intra or Inter subject data splitting
def split_data(data, split_type='intra'):
    if split_type == 'intra':
        subject_ids = data['subject_id'].unique()
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        for subject_id in subject_ids:
            subject_data = data[data['subject_id'] == subject_id]
            split_index = int(0.8 * len(subject_data))
            train_data = pd.concat([train_data, subject_data.iloc[:split_index]])
            test_data = pd.concat([test_data, subject_data.iloc[split_index:]])

        return train_data, test_data

    elif split_type == 'inter':
        subject_ids = data['subject_id'].unique()

        # Randomly select
        random.shuffle(subject_ids)
        train_subjects = subject_ids[:9]
        test_subjects = subject_ids[9:]

        train_data = data[data['subject_id'].isin(train_subjects)]
        test_data = data[data['subject_id'].isin(test_subjects)]

        return train_data, test_data

    else:
        raise ValueError("Invalid split_type. Choose 'intra' or 'inter'.")


# Function to normalize data to the range [0, 1]
def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data[['time', 'angle']] = scaler.fit_transform(data[['time', 'angle']])
    return data


# Function to reshape data for CNN-BiLSTM and MLP models
def reshape_data(train_data, test_data, amount_of_features):
    X_train_cnn = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], amount_of_features))
    X_test_cnn = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], amount_of_features))

    X_train_mlp = train_data.reshape(train_data.shape[0], -1)
    X_test_mlp = test_data.reshape(test_data.shape[0], -1)

    return X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp


def process_data(input_file, output_file, target_count, amount_of_features, split_type='intra'):
    data = pd.read_csv(input_file)

    if 'time' not in data.columns or 'angle' not in data.columns or 'subject_id' not in data.columns:
        raise ValueError("Input CSV must contain 'time', 'angle', and 'subject_id' columns.")

    # Assign labels to the data based on gait phases
    data = assign_labels_to_gait_phases(data)

    # Encode labels to one-hot encoding
    labels_one_hot = encode_labels_to_one_hot(data)

    # Compute the label proportions and target counts
    label_proportions, target_counts, alpha, new_samples_count = compute_label_proportions(data)
    print("Label Proportions:", label_proportions)
    print("Target Counts:", target_counts)
    print(f"Alpha: {alpha}, New Samples Count: {new_samples_count}")

    # Generate new data for each label based on its target count
    augmented_data = pd.DataFrame()
    for label, target in target_counts.items():
        label_data = data[data['label'] == label]
        label_target_count = int(target)
        augmented_label_data = apply_spline_balance(label_data, label_target_count)
        augmented_label_data['label'] = label
        augmented_data = pd.concat([augmented_data, augmented_label_data], ignore_index=True)

    # Split data into training and test sets based on intra or inter splitting
    train_data, test_data = split_data(augmented_data, split_type)

    # Normalize data to the range [0, 1]
    normalize_train_data = normalize_data(train_data)
    normalize_test_data = normalize_data(test_data)

    # Reshape data for CNN-BiLSTM and MLP models
    X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp = reshape_data(normalize_train_data, normalize_test_data,
                                                                    amount_of_features)

    # Save the augmented, normalized, and reshaped data to new CSV files
    train_output_file = output_file.replace('.csv', '_train.csv')
    test_output_file = output_file.replace('.csv', '_test.csv')
    train_data.to_csv(train_output_file, index=False)
    test_data.to_csv(test_output_file, index=False)

    print(f"Training data saved to {train_output_file}")
    print(f"Test data saved to {test_output_file}")

    # Plot processed data
    plt.figure(figsize=(15, 6))
    plt.plot(data['time'], data['angle'], label='True', color='blue')
    plt.plot(augmented_data['time'], augmented_data['angle'], label='Prediction', linestyle='dashed', color='orange')
    plt.xlabel('Time Steps')
    plt.ylabel('Joint Angles')
    plt.legend()
    plt.show()

    return X_train_cnn, X_test_cnn, X_train_mlp, X_test_mlp, \
           normalize_train_data, normalize_test_data, labels_one_hot

# y_train, y_test = normalize_train_data, normalize_test_data

