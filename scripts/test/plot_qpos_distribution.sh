#!/bin/bash
source activate anypos
python utils/test/plot_qpos_distribution.py --save_dir output/ --data1 <data_1 dir> --data2 <data_2 dir> --label1 'Test Dataset' --label2 'Task-Agnostic Random Actions'