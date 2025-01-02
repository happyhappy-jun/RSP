import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Epoch': [400, 200, 300, 200, 300, 400, 100, 200, 300, 200, 300, 200, 400, 400, 300],
    'kl_scale': [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001, 0.01, 0.005, 0.001],
    'JF_Mean': [60.1, 57.9, 58.7, 56.9, 58.1, 58.1, 55.7, 56.8, 58.6, 57.3, 57.8, 57.6, 59.0, 58.7, 58.5],
    'J_Mean': [57.4, 55.3, 55.9, 53.8, 55.2, 55.2, 52.9, 54.0, 56.0, 54.7, 55.1, 55.0, 56.2, 56.0, 55.9],
    'F_Mean': [62.8, 60.7, 61.5, 59.9, 60.9, 61.0, 58.4, 59.6, 61.3, 60.0, 60.6, 60.4, 61.9, 61.3, 61.1],
    'Model': ['Baseline(RSP)', 'RSP-Fixed', 'RSP-Fixed',
              'RSP-LLM-Fixed', 'RSP-LLM-Fixed', 'RSP-LLM-Fixed',
              'RSP-LLM-Fixed', 'RSP-LLM-Fixed', 'RSP-LLM-Fixed',
              'RSP-LLM-Fixed', 'RSP-LLM-Fixed', 'RSP-LLM-Fixed-Cosine',
              'RSP-Fixed', 'RSP-LLM-Fixed', 'RSP-LLM-Fixed-Cosine']
}
df = pd.DataFrame(data)

# Set custom color palette with more vivid colors
colors = ['#FF1E1E', '#00CC00', '#0066FF']  # Brighter Red, Green, Blue
sns.set_palette(colors)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
sns.set_style("white")

# Create scatter-line plots with larger markers and thicker lines
scatter = sns.scatterplot(data=df, x='Epoch', y='JF_Mean', hue='kl_scale', style='Model', s=150, alpha=1)
lines = sns.lineplot(data=df, x='Epoch', y='JF_Mean', hue='kl_scale', style='Model', linewidth=2.5)

# Set y-axis limits to better show the differences
plt.ylim(55, 61)

plt.title('Model Performance across Epochs', fontsize=14, pad=20)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('J&F Mean Score', fontsize=12)

# Format legend
plt.legend(title='KL Scale', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True,
          edgecolor='black', fancybox=True, shadow=True)

plt.gca().set_facecolor('white')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('davis_jfmean.png', dpi=300)
plt.show()