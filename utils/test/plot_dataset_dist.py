import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
# Note:
# set the path of the two datasets
# change the fig labels accordingly


def load_qpos(dataset_path):
    qposes = torch.tensor([])
    for task_name in os.listdir(dataset_path):
        task_path = os.path.join(dataset_path, task_name)
        if not os.path.isdir(task_path):
            continue
        for file_name in os.listdir(task_path):
            if file_name.endswith('.pt'):
                qpos_path = os.path.join(task_path, file_name)
                # Load qpos
                try:
                    qpos = torch.load(qpos_path)
                    qposes = torch.cat((qposes, qpos), dim=0)
                except:
                    print(f"Skipping {qpos_path} - failed to load qpos")
                    continue
    return qposes


def main():
    if len(sys.argv) < 3:
        qposes = [load_qpos(sys.argv[1])]
    else:
        qposes = [load_qpos(sys.argv[1]), load_qpos(sys.argv[2])]
    num_dimensions = qposes[0].shape[1]

    # use density distribution
    # use_density = False

    fig, axs = plt.subplots(2, 7, figsize=(30, 10))

    for i in range(num_dimensions):
        ax1 = axs[i // 7, i % 7]
        if i == 0:
            ax1.hist(qposes[0][:, i], bins=100, alpha=0.5, label=sys.argv[1].split('/')[-1], color='blue', density=True)
        else:
            ax1.hist(qposes[0][:, i], bins=100, alpha=0.5, color='blue', density=True)
        # ax1.legend(loc='upper left')
        ax1.set_title(f'Dimension {i}')
        if len(qposes) > 1:
            ax2 = ax1.twinx()
            if i == 0:
                ax2.hist(qposes[1][:, i], bins=100, alpha=0.5, label=sys.argv[2].split('/')[-1], color='orange', density=True)
            else:
                ax2.hist(qposes[1][:, i], bins=100, alpha=0.5, color='orange', density=True)
            # ax2.legend(loc='upper right')
    fig.legend()
        
    # fig.suptitle('density distribution' if use_density else 'frequency distribution', fontsize=16)
    plt.tight_layout()
    plt.savefig('dataset_dist.png')


if __name__ == "__main__":
    main()
