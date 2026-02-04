"""
CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
Paper: https://openreview.net/forum?id=86IvZmY26S
Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
         Zhentao Tang, Mingxuan Yuan, Junchi Yan
License: Non-Commercial License (see LICENSE). Commercial use requires permission.
Signature: CORE Authors (NeurIPS 2025)
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
    with open(path,'r') as f:
        res = json.load(f)
    return res


def save_json(obj, path:str):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4)


def load_pkl(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
        return res


def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def setup_seed(seed = 3407):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    # https://zhuanlan.zhihu.com/p/73711222
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_datetime():
    t = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    return t


class Logger():
    def __init__(self,log_file_path) -> None:
        self.path = log_file_path
        with open(self.path,'w') as f:
            f.write(get_datetime() + "\n")
            print(get_datetime())
        return
    
    def log(self,content):
        content = str(content)
        with open(self.path,'a') as f:
            f.write(content + "\n")
            print(content)
        return


def mkdir(dir, rm=False):
    if os.path.isdir(dir):
        if rm:
            shutil.rmtree(dir)
            os.makedirs(dir)
        else:
            pass
    else:
        os.makedirs(dir)


def convert_dict_to_args(d):
    parser = argparse.ArgumentParser()
    for k,v in d.items():
        parser.add_argument(f'--{k}', default=v)
    return parser.parse_args()


def record_time(func):
    @functools.wraps(func)
    def run(*args, **kwds):
        torch.cuda.synchronize()
        s = time.time()
        ans = func(*args, **kwds)
        torch.cuda.synchronize()
        e = time.time()
        print(f"Running time for {func.__name__} = {e-s} (s)")
        return ans
    return run


def draw(tml_list, new_x, new_y, ourwandb, total_steps, result_file:str, output_path:str, num_layer:int, *emphasize):
    fig = plt.figure()
    border_max = -1
    num_row = int(num_layer**0.5)
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
            width.append(float(s[3])-float(s[1]))
            height.append(float(s[4])-float(s[2]))


    for target_layer in range(num_layer):
        ax = fig.add_subplot(num_row,num_col,target_layer+1)
        for x,y,w,h,n,o,l in zip(x_cor, y_cor, width, height, name, order, layer):
            if l == target_layer:
                rect1 = matplotlib.patches.Rectangle((x, y), w, h, edgecolor="black", fill=True, alpha=.3)
                ax.add_patch(rect1)
                # ax.text(x+w//2, y+h//2, str(o), horizontalalignment='center', verticalalignment='center')

        ax.set_aspect('equal', adjustable='box')
        border_max = max(record['x_max'],record['y_max'],border_max)
        plt.xlim([0, border_max+200])
        plt.ylim([0, border_max+200])
        plt.title('cost = {:.3f}, HPWL = {:.3f}\narea = {:.3f}, feedthrough = {:.3f}'.format(record['cost'], record['wirelength'], record['area'], record['feedthrough']))

    horizontal_line = [(0, new_y), (new_x, new_y)]  # 横线，起点 (0, 0)，终点 (1, 0)
    vertical_line = [(new_x, 0), (new_x, new_y)]  # 竖线，起点 (0.5, -0.5)，终点 (0.5, 0.5)

    # 提取坐标
    h_x, h_y = zip(*horizontal_line)
    v_x, v_y = zip(*vertical_line)
    plt.plot(h_x, h_y, color='blue')
    plt.plot(v_x, v_y, color='blue')

    x, y = zip(*tml_list)
    plt.plot(x, y, 'o', markersize=1.0, color='red')
    #plt.scatter(x, y, s=0.1 ,color='red', label='Data Points')

    plt.savefig(output_path)
    #plt.close()
    if ourwandb is not None:
        #print("save plt ...")
        ourwandb.log({"Best_FP_plt": wandb.Image(output_path)}, step=total_steps)
        #wandb.log({"Best_FP_plt": wandb.Image(output_path, caption="epoch:{}".format(total_steps))})
    plt.close()


def draw_string(tml_list, new_x, new_y, ourwandb, index, total_steps, result_file:str, output_path:str, num_layer:int, *emphasize):
    fig = plt.figure()
    border_max = -1
    num_row = int(num_layer**0.5)
    num_col = math.ceil(num_layer // num_row)

    x_cor = []
    y_cor = []
    width = []
    height = []
    name = []
    order = []
    layer = []

    lines = result_file.split('\n')





    pattern = re.compile(r'(\w+)\s*=\s*([+-]?\d*\.?\d+)')
    record = {}
    for _ in range(7):
       # print(lines[_])
        line = lines[_]
        r = pattern.search(line)
        record[r.group(1)] = float(r.group(2))
       # print(r.group(1), "->", r.group(2))

    for line in  lines[7:]:
    #    print(line)
        if line == '':
            continue
        line = line.strip()
        s = line.split(' ')
        layer.append(int(s[5]))
        name.append(str(s[0]))
        order.append(int(s[6]))
        x_cor.append(float(s[1]))
        y_cor.append(float(s[2]))
        width.append(float(s[3])-float(s[1]))
        height.append(float(s[4])-float(s[2]))
   # assert 1 == 2

    for target_layer in range(num_layer):
        ax = fig.add_subplot(num_row,num_col,target_layer+1)
        for x,y,w,h,n,o,l in zip(x_cor, y_cor, width, height, name, order, layer):
            if l == target_layer:
                rect1 = matplotlib.patches.Rectangle((x, y), w, h, edgecolor="black", fill=True, alpha=.3)
                ax.add_patch(rect1)
                # ax.text(x+w//2, y+h//2, str(o), horizontalalignment='center', verticalalignment='center')

        ax.set_aspect('equal', adjustable='box')
        border_max = max(record['x_max'],record['y_max'],border_max)
        plt.xlim([0, border_max+200])
        plt.ylim([0, border_max+200])
        plt.title('cost = {:.3f}, HPWL = {:.3f}\narea = {:.3f}, feedthrough = {:.3f}'.format(record['cost'], record['wirelength'], record['area'], record['feedthrough']))

    horizontal_line = [(0, new_y), (new_x, new_y)]  # 横线，起点 (0, 0)，终点 (1, 0)
    vertical_line = [(new_x, 0), (new_x, new_y)]  # 竖线，起点 (0.5, -0.5)，终点 (0.5, 0.5)

    # 提取坐标
    h_x, h_y = zip(*horizontal_line)
    v_x, v_y = zip(*vertical_line)
    plt.plot(h_x, h_y, color='blue')
    plt.plot(v_x, v_y, color='blue')

    x, y = zip(*tml_list)
    plt.plot(x, y, 'o', markersize=1.0, color='red')
    #plt.scatter(x, y, s=0.1 ,color='red', label='Data Points')

    plt.savefig(output_path)
    #plt.close()
    if ourwandb is not None:
        #print("save plt ...")

     #   print("?????????????", total_steps)
        ourwandb.log({"Best_FP_plt_" + str(index): wandb.Image(output_path)}, step=total_steps)
        #wandb.log({"Best_FP_plt": wandb.Image(output_path, caption="epoch:{}".format(total_steps))})
    plt.close()




def draw_class(class_num, tml_list, new_x, new_y, ourwandb, total_steps, result_file:str, output_path:str, num_layer:int, *emphasize):
    fig = plt.figure()
    border_max = -1
    num_row = int(num_layer**0.5)
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
            width.append(float(s[3])-float(s[1]))
            height.append(float(s[4])-float(s[2]))


    for target_layer in range(num_layer):
        ax = fig.add_subplot(num_row,num_col,target_layer+1)
        for x,y,w,h,n,o,l in zip(x_cor, y_cor, width, height, name, order, layer):
            if l == target_layer:
                rect1 = matplotlib.patches.Rectangle((x, y), w, h, edgecolor="black", fill=True, alpha=.3)
                ax.add_patch(rect1)
                # ax.text(x+w//2, y+h//2, str(o), horizontalalignment='center', verticalalignment='center')

        ax.set_aspect('equal', adjustable='box')
        border_max = max(record['x_max'],record['y_max'],border_max)
        plt.xlim([0, border_max+200])
        plt.ylim([0, border_max+200])
        plt.title('cost = {:.3f}, HPWL = {:.3f}\narea = {:.3f}, feedthrough = {:.3f}'.format(record['cost'], record['wirelength'], record['area'], record['feedthrough']))

    horizontal_line = [(0, new_y), (new_x, new_y)]  # 横线，起点 (0, 0)，终点 (1, 0)
    vertical_line = [(new_x, 0), (new_x, new_y)]  # 竖线，起点 (0.5, -0.5)，终点 (0.5, 0.5)

    # 提取坐标
    h_x, h_y = zip(*horizontal_line)
    v_x, v_y = zip(*vertical_line)
    plt.plot(h_x, h_y, color='blue')
    plt.plot(v_x, v_y, color='blue')

    x, y = zip(*tml_list)
    plt.plot(x, y, 'o', markersize=1.0, color='red')
    #plt.scatter(x, y, s=0.1 ,color='red', label='Data Points')

    plt.savefig(output_path)
    #plt.close()
    if ourwandb is not None:
        #print("save plt ...")
        ourwandb.log({"Best_" + str(class_num)+ "_FP_plt": wandb.Image(output_path)}, step=total_steps)
        #wandb.log({"Best_FP_plt": wandb.Image(output_path, caption="epoch:{}".format(total_steps))})
    plt.close()

from sklearn.manifold import TSNE
#import umap
def draw_tsne(data, n_clusters, labels, output_path, ourwandb, total_steps):
    fig = plt.figure()
    border_max = -1

    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(np.array(data))
    #umap_model = umap.UMAP(n_components=2, random_state=42)
    #reduced_data = umap_model.fit_transform(data)


    # 根据聚类标签绘制不同颜色的散点
    for label in range(n_clusters):
        indices = np.where(labels == label)
        plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], label=f'Cluster {label}')

    plt.title('UMAP Visualization of Clusters')
    plt.legend()
    #plt.scatter(x, y, s=0.1 ,color='red', label='Data Points')

    plt.savefig(output_path)
    #plt.close()
    if ourwandb is not None:
        #print("save plt ...")
        ourwandb.log({"UMAP": wandb.Image(output_path)}, step=total_steps)
        #wandb.log({"Best_FP_plt": wandb.Image(output_path, caption="epoch:{}".format(total_steps))})
    plt.close()


if __name__ == "__main__":
    print(get_datetime())