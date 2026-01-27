"""
Utility functions for CORE floorplanning.
"""

import random
import json
import numpy as np
import os
import torch
import pickle
import time
import argparse
import shutil
import functools
import matplotlib
import matplotlib.pyplot as plt
import math
import re
import wandb

matplotlib.use('Agg')


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def save_json(obj, path):
    """Save object to JSON file."""
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4)


def load_pkl(path):
    """Load pickle file."""
    with open(path, 'rb') as f:
        res = pickle.load(f)
    return res


def save_pkl(obj, path):
    """Save object to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def setup_seed(seed=3407):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_datetime():
    """Get current datetime string."""
    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    return t


class Logger:
    """Simple file logger."""

    def __init__(self, log_file_path):
        self.path = log_file_path
        with open(self.path, 'w') as f:
            f.write(get_datetime() + "\n")
            print(get_datetime())

    def log(self, content):
        content = str(content)
        with open(self.path, 'a') as f:
            f.write(content + "\n")
            print(content)


def mkdir(dir, rm=False):
    """Create directory, optionally removing existing one."""
    if os.path.isdir(dir):
        if rm:
            shutil.rmtree(dir)
            os.makedirs(dir)
    else:
        os.makedirs(dir)


def convert_dict_to_args(d):
    """Convert dictionary to argparse namespace."""
    parser = argparse.ArgumentParser()
    for k, v in d.items():
        parser.add_argument(f'--{k}', default=v)
    return parser.parse_args()


def record_time(func):
    """Decorator to record function execution time."""
    @functools.wraps(func)
    def run(*args, **kwds):
        torch.cuda.synchronize()
        s = time.time()
        ans = func(*args, **kwds)
        torch.cuda.synchronize()
        e = time.time()
        print(f"Running time for {func.__name__} = {e - s} (s)")
        return ans
    return run


def draw(tml_list, new_x, new_y, ourwandb, total_steps, result_file, output_path, num_layer, *emphasize):
    """Draw floorplan visualization."""
    fig = plt.figure()
    border_max = -1
    num_row = int(num_layer ** 0.5)
    num_col = math.ceil(num_layer // num_row)

    x_cor = []
    y_cor = []
    width = []
    height = []
    name = []
    order = []
    layer = []

    with open(result_file) as f:
        pattern = re.compile(r'(\w+)\s*=\s*([+-]?\d*\.?\d+)')
        record = {}
        for _ in range(7):
            line = f.readline()
            r = pattern.search(line)
            record[r.group(1)] = float(r.group(2))

        for line in f.readlines():
            line = line.strip()
            s = line.split(' ')
            layer.append(int(s[5]))
            name.append(str(s[0]))
            order.append(int(s[6]))
            x_cor.append(float(s[1]))
            y_cor.append(float(s[2]))
            width.append(float(s[3]) - float(s[1]))
            height.append(float(s[4]) - float(s[2]))

    for target_layer in range(num_layer):
        ax = fig.add_subplot(num_row, num_col, target_layer + 1)
        for x, y, w, h, n, o, l in zip(x_cor, y_cor, width, height, name, order, layer):
            if l == target_layer:
                rect1 = matplotlib.patches.Rectangle(
                    (x, y), w, h, edgecolor="black", fill=True, alpha=.3
                )
                ax.add_patch(rect1)

        ax.set_aspect('equal', adjustable='box')
        border_max = max(record['x_max'], record['y_max'], border_max)
        plt.xlim([0, border_max + 200])
        plt.ylim([0, border_max + 200])
        plt.title(
            f"cost = {record['cost']:.3f}, HPWL = {record['wirelength']:.3f}\n"
            f"area = {record['area']:.3f}, feedthrough = {record['feedthrough']:.3f}"
        )

    # Draw chip boundary
    horizontal_line = [(0, new_y), (new_x, new_y)]
    vertical_line = [(new_x, 0), (new_x, new_y)]

    h_x, h_y = zip(*horizontal_line)
    v_x, v_y = zip(*vertical_line)
    plt.plot(h_x, h_y, color='blue')
    plt.plot(v_x, v_y, color='blue')

    # Draw terminals
    x, y = zip(*tml_list)
    plt.plot(x, y, 'o', markersize=1.0, color='red')

    plt.savefig(output_path)
    
    if ourwandb is not None:
        ourwandb.log({"Best_FP_plt": wandb.Image(output_path)}, step=total_steps)
    plt.close()


if __name__ == "__main__":
    print(get_datetime())
