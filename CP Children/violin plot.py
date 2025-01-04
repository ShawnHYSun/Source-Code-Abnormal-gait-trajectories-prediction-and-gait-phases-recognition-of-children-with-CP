import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Data
# Input the statistical results
data = []

# Calculate mean and standard deviation
mean_value = np.mean(data)
std_dev = np.std(data)
print(f'mean_value={mean_value:.2f}')
print(f'std_dev={std_dev:.2f}')

# Calculate mean 1SD
# mean_minus_1sd = mean_value - std_dev
# mean_plus_1sd = mean_value + std_dev

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create figure with adjusted aspect ratio (narrower and slightly taller)
plt.figure(figsize=(3, 6))

# Plot half violin plot with individual data points as dots
# Green: 217/255, 242/255, 208/255 -- CL
# Pink: 242/255, 207/255, 238/255 -- CLA
# Blue: 202/255, 238/255, 251/255 -- CLMA
# Brown: 251/255, 227/255, 214/255 -- Ours

sns.violinplot(data=data, inner="box", split=False, scale='width',
               linewidth=1.5, bw=0.5, orient='v', color=(251/255, 227/255, 214/255))

# Add horizontal lines for mean and 1SD
# plt.axhline(mean_value, color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_value:.3f}')
# plt.axhline(mean_minus_1sd, color='blue', linestyle='--', linewidth=1, label=f'Mean-1SD: {mean_minus_1sd:.3f}')
# plt.axhline(mean_plus_1sd, color='red', linestyle='--', linewidth=1, label=f'Mean+1SD: {mean_plus_1sd:.3f}')

# Add labels for mean 1SD
# plt.text(0, mean_value + 0.005, f'{mean_value:.3f}', color='black', fontsize=10, fontweight='bold')
# plt.text(0, mean_minus_1sd + 0.005, f'{mean_minus_1sd:.3f}', color='blue', fontsize=10, fontweight='bold')
# plt.text(0, mean_plus_1sd + 0.005, f'{mean_plus_1sd:.3f}', color='red', fontsize=10, fontweight='bold')

# Adjust axis tick font size and boldness
plt.xticks(fontsize=14, fontweight='bold')  # Adjust X-axis tick labels
plt.yticks(fontsize=14, fontweight='bold')  # Adjust Y-axis tick labels

# Set title and adjust plot limits  MAE(0, 0.3) R2 (0,1)
plt.ylim(0, 1)
# plt.title('MAE', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()