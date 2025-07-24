import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime
from sklearn.manifold import TSNE

def load_qposes(dataset_path):
    qposes = torch.tensor([])
    for task_name in os.listdir(dataset_path):
        task_path = os.path.join(dataset_path, task_name)
        if not os.path.isdir(task_path):
            continue
        for file_name in os.listdir(task_path):
            if file_name.endswith('.mp4'):
                episode_idx = file_name.split('_')[1].split('.')[0]
                video_path = os.path.join(task_path, file_name)
                qpos_path = os.path.join(task_path, f'episode_{episode_idx}_qpos.pt')
                
                if not os.path.exists(qpos_path):
                    print(f"Skipping {video_path} - no matching qpos file")
                    continue
                # Load qpos
                qpos = torch.load(qpos_path)
                qposes = torch.cat((qposes, qpos), dim=0)
    return qposes
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
def plot_distribution(qposes_1, qposes_2, label1, label2, save_dir, use_density=False):
    num_dimensions = qposes_1.shape[1]
    fig, axs = plt.subplots(4, 4, figsize=(90, 50))
    plt.rcParams['font.size'] = 110
    row_array=[0,0,0,0,1,1,1,2,2,2,2,3,3,3]
    col_array=[0,1,2,3,0,1,2,0,1,2,3,0,1,2]
    for i in range(num_dimensions):
        # row = i // 4
        # col = i % 4
        ax1 = axs[row_array[i], col_array[i]]
        ax2 = ax1.twinx()
        ax1.hist(qposes_1[:, i], bins=100, alpha=0.5, color='blue', density=use_density)
        ax2.hist(qposes_2[:, i], bins=100, alpha=0.5, color='pink', density=use_density)
        # ax1.legend(handles=legend_elements, loc='upper left', fontsize=90, prop=font_prop)
        ax1.set_title(f'Dimension {i}', fontsize=110)
    fig.delaxes(axs[1, 3])
    fig.delaxes(axs[3, 3])
    legend_elements = [
        Line2D([0], [0], color='blue', lw=8, label='dataset_1' if label1 is None else label1),
        Line2D([0], [0], color='pink', lw=8, label='dataset_2' if label2 is None else label2)
    ]
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.99, 0.03), fontsize=110)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"qpos_distribution_{timestamp}.pdf")
    plt.savefig(save_path, dpi=600, format='pdf')
    
def main():
    parser = argparse.ArgumentParser(description="Plot qpos distribution from two datasets.")
    parser.add_argument("--data1", type=str, help="Path to the first dataset")
    parser.add_argument("--data2", type=str, help="Path to the second dataset")
    parser.add_argument("--label1", type=str, default=None, help="Label to the first dataset")
    parser.add_argument("--label2", type=str, default=None, help="Label to the second dataset")
    parser.add_argument("--use_density", action="store_true", help="Use density distribution instead of frequency")
    parser.add_argument("--save_dir", type=str, default='output', help="Path to save the figure (optional)")
    args = parser.parse_args()
    
    print(args)
    
    print('loading qpos...')
    qposes_1 = load_qposes(args.data1)
    qposes_2 = load_qposes(args.data2)
    print('qpos loaded')
    
    print('ploting...')
    plot_distribution(qposes_1, qposes_2, args.label1, args.label2, args.save_dir, args.use_density)
    print('Success!')

if __name__ == "__main__":
    main()