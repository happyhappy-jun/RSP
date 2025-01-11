import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle, Patch

# Set the style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create the data
data = [
    {"model": "Baseline(RSP)", "epoch": 400, "kl_scale": 0.01, "jf_mean": 60.1},
    {"model": "RSP-Fixed", "epoch": 200, "kl_scale": 0.01, "jf_mean": 57.9},
    {"model": "RSP-Fixed", "epoch": 300, "kl_scale": 0.01, "jf_mean": 58.7},
    {"model": "RSP-Fixed", "epoch": 400, "kl_scale": 0.01, "jf_mean": 59.0},
    {"model": "RSP-LLM-Fixed", "epoch": 200, "kl_scale": 0.01, "jf_mean": 56.9},
    {"model": "RSP-LLM-Fixed", "epoch": 300, "kl_scale": 0.01, "jf_mean": 58.1},
    {"model": "RSP-LLM-Fixed", "epoch": 400, "kl_scale": 0.01, "jf_mean": 58.1},
    {"model": "RSP-LLM-Fixed", "epoch": 200, "kl_scale": 0.005, "jf_mean": 56.8},
    {"model": "RSP-LLM-Fixed", "epoch": 300, "kl_scale": 0.005, "jf_mean": 58.6},
    {"model": "RSP-LLM-Fixed", "epoch": 400, "kl_scale": 0.005, "jf_mean": 58.7},
    {"model": "RSP-LLM-Fixed", "epoch": 200, "kl_scale": 0.001, "jf_mean": 57.3},
    {"model": "RSP-LLM-Fixed", "epoch": 300, "kl_scale": 0.001, "jf_mean": 57.8},
    {"model": "RSP-LLM-Fixed Cosine", "epoch": 200, "kl_scale": 0.001, "jf_mean": 57.6},
    {"model": "RSP-LLM-Fixed Cosine", "epoch": 300, "kl_scale": 0.001, "jf_mean": 58.5},
    {"model": "RSP-LLM-Fixed Cosine", "epoch": 400, "kl_scale": 0.001, "jf_mean": 58.8},
    {"model": "RSP-LLM-Fixed Cosine", "epoch": 400, "kl_scale": 0.005, "jf_mean": 58.9},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Filter data for epoch 400
df_400 = df[df['epoch'] == 400].copy()

# Create combined model name with KL scale
df_400['model_kl'] = df_400.apply(lambda x: f"{x['model']}\n(KL={x['kl_scale']})", axis=1)

# Reset index to ensure proper indexing
df_400 = df_400.reset_index(drop=True)

# Define colors for each model type
color_map = {
    'Baseline(RSP)': '#1f77b4',
    'RSP-Fixed': '#2ca02c',
    'RSP-LLM-Fixed': '#ff7f0e',
    'RSP-LLM-Fixed Cosine': '#d62728'
}

# Create color list based on model names
colors = [color_map[model] for model in df_400['model']]

# Create hatch patterns based on KL scale
hatch_map = {
    0.01: '',  # No pattern
    0.005: '///',  # Diagonal lines
    0.001: '...'  # Dots
}

# Create the plot
fig = plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_400,
                 x='model_kl',
                 y='jf_mean',
                 palette=colors,
                 width=0.7)

# Add hatches based on KL scale
for i, bar in enumerate(ax.patches):
    kl_scale = df_400.iloc[i]['kl_scale']
    bar.set_hatch(hatch_map[kl_scale])

# Customize the plot
plt.ylabel('JF Mean Score', fontsize=12)
plt.xlabel('')  # Remove x-label as it's redundant
plt.title('Model Performance Comparison at Epoch 400', fontsize=14, pad=20)

# Set y-axis limits to focus on relevant range
plt.ylim(55, 61)

# Add break symbol on y-axis
d = 0.01  # Size of the break symbol
kwargs = dict(transform=ax.transData, color='k', clip_on=False, lw=1)
ax.plot((-d, +d), (55, 55 - d), **kwargs)
ax.plot((-d, +d), (55 - 1.5 * d, 55 - 2.5 * d), **kwargs)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

# Add value labels on top of each bar
for i, bar in enumerate(ax.patches):
    height = bar.get_height()
    is_final_model = (df_400.iloc[i]['model'] == 'RSP-LLM-Fixed Cosine' and
                      df_400.iloc[i]['kl_scale'] == 0.005)
    font_weight = 'bold' if is_final_model else 'normal'

    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.1f}',
            ha='center', va='bottom',
            weight=font_weight)

# Add star marker for final model
final_model_mask = (df_400['model'] == 'RSP-LLM-Fixed Cosine') & (df_400['kl_scale'] == 0.005)
final_model_idx = df_400.index[final_model_mask][0]
final_bar = ax.patches[final_model_idx]
plt.plot(final_model_idx, final_bar.get_height() + 0.3,
         marker='*', color='#d62728', markersize=20,
         markeredgecolor='black', markeredgewidth=1)

# Create legend elements
# KL scale legend
kl_legend_elements = [Rectangle((0, 0), 1, 1, facecolor='gray',
                                hatch=pattern, label=f'KL={scale}')
                      for scale, pattern in hatch_map.items()]

# Model type legend
model_legend_elements = [Patch(facecolor=color, label=model)
                         for model, color in color_map.items()]

# Combine both legends
all_legend_elements = model_legend_elements + kl_legend_elements

# Create two-column legend
plt.legend(handles=all_legend_elements,
           ncol=1,  # Two columns
           bbox_to_anchor=(1.0, 1),  # Position outside the plot
           loc='upper left',
           title='Models and KL Scales')

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()