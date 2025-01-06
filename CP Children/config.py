
# General
window = 5
lstm_units = 16
mlp_units = 32
dropout = 0.2
epoch = 70
batch_size = 256
learning_rate = 0.01
amount_of_features = 2
loss_prediction = 'mse'

# Dataset
input_file = 'path_to_input_file.csv'
output_file = 'path_to_output_file.csv'
target_count = 50000  # Target number of time steps in the augmented dataset. 50000 is just an example.
split_type = 'intra'  # 'intra' or 'inter'
