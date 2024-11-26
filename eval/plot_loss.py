import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_latest_event_file(directory):
    """Get the most recent event file in the directory."""
    event_files = [f for f in os.listdir(directory) if f.startswith('events.out.tfevents')]
    if not event_files:
        return None
    event_files.sort(key=lambda x: float(x.split('.')[3]))
    return os.path.join(directory, event_files[-1])

def load_tensorboard_data(path, tag):
    """Load data from tensorboard log file."""
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        return steps, values
    return None, None

def ema_smooth(values, alpha=0.95):
    """Apply exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * values[i]
    return smoothed

def plot_all_metrics(base_dir, metrics, alpha=0.99, figsize=(15, 12)):
    """Create subplots for each metric."""
    variants = ['rsp', 'rsp-fix', 'rsp-llm-fix', 'rsp-llm-global']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Training Metrics Comparison (EMA α=0.99)', fontsize=16, y=0.95)
    
    axes_flat = axes.flatten()
    
    for metric_idx, metric in enumerate(metrics):
        ax = axes_flat[metric_idx]
        
        for variant_idx, variant in enumerate(variants):
            directory = os.path.join(base_dir, variant)
            if not os.path.exists(directory):
                continue
                
            latest_event_file = get_latest_event_file(directory)
            if latest_event_file is None:
                continue
                
            steps, values = load_tensorboard_data(latest_event_file, metric)
            
            if steps is None:
                continue
                
            steps = np.array(steps)
            values = np.array(values)
            smoothed_values = ema_smooth(values, alpha)
            
            ax.plot(steps, smoothed_values, label=variant, color=colors[variant_idx], linewidth=2)
        
        ax.set_title(f'{metric}', fontsize=12)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
    # Remove any extra subplots
    for idx in range(n_metrics, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    plt.tight_layout()
    return fig

def plot_individual_metrics(base_dir, metrics, alpha=0.99):
    """Create individual plots for each metric."""
    variants = ['rsp', 'rsp-fix', 'rsp-llm-fix', 'rsp-llm-global']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for metric in metrics:
        plt.figure(figsize=(6, 4))
        
        for variant_idx, variant in enumerate(variants):
            directory = os.path.join(base_dir, variant)
            if not os.path.exists(directory):
                continue
                
            latest_event_file = get_latest_event_file(directory)
            if latest_event_file is None:
                continue
                
            steps, values = load_tensorboard_data(latest_event_file, metric)
            
            if steps is None:
                continue
                
            steps = np.array(steps)
            values = np.array(values)
            smoothed_values = ema_smooth(values, alpha)
            
            plt.plot(steps, smoothed_values, label=variant, color=colors[variant_idx], linewidth=2)
        
        plt.title(f'{metric} During Training')
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(f'{metric}_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

# Base directory path
base_dir = '/home/junyoon/rsp-llm/RSP/logs'

# Metrics to plot
metrics_to_plot = ['train_loss', 'loss_post', 'loss_prior', 'loss_kl', 'loss_mae']

# Create combined plot
fig = plot_all_metrics(
    base_dir,
    metrics_to_plot,
    alpha=0.95,
    figsize=(9, 6)
)

# Save combined plot
plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# Create individual plots
# plot_individual_metrics(base_dir, metrics_to_plot, alpha=0.99)