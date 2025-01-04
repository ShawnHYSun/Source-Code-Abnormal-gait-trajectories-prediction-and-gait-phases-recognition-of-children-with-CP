
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
input_file = 'C:/Users/15308/BaiduSyncdisk/SCI Paper/2nd version/c1c10/C1/C1-LHF.csv'
output_file = 'path_to_output_file.csv path_to_input_file.csv'
target_count = 40000  # Target number of time steps in the augmented dataset
split_type = 'intra'  # 'intra' or 'inter'
