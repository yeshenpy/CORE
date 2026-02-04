"""
CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
Paper: https://openreview.net/forum?id=86IvZmY26S
Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
         Zhentao Tang, Mingxuan Yuan, Junchi Yan
License: Non-Commercial License (see LICENSE). Commercial use requires permission.
Signature: CORE Authors (NeurIPS 2025)
"""

import tree
import time
import os
import numpy as np
import argparse
import utils
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
# NOTE: Do not hardcode secrets in source code.
# If you want to use Weights & Biases, set `WANDB_API_KEY` in your environment.
os.environ.setdefault("WANDB_MODE", "offline")
torch.set_num_threads(cpu_num)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=str, default="n300")
    parser.add_argument("--enable_draw", type=int, default=1)
    parser.add_argument("--result_dir", type=str, default=f"./result-SA")
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--gap_iter_update_temperature", type=int, default=1000)
    parser.add_argument("--weight_hpwl", type=float, default=0.0)
    parser.add_argument("--weight_area", type=float, default=0.5)
    parser.add_argument("--weight_feedthrough", type=float, default=0)
    parser.add_argument("--init_temperature", type=float, default=1e6)

    args = parser.parse_args()

    args.result_dir = os.path.join( args.result_dir, args.circuit )
    return args


def main(input_blk:str, input_net:str, output, png_path, enable_draw:bool, num_layer, gap_iter_update_temperature, result_dir:str, weight_hpwl, weight_area, weight_feedthrough, init_temperature):
    s = time.time()
    df = pd.DataFrame()
    csv_path = os.path.join( result_dir, "record.csv" )
    fp = tree.Floorplanner(
        num_layer, 
        weight_hpwl,
        weight_area,
        weight_feedthrough,
    )
    fp.parse_blk(input_blk)
    fp.parse_net(input_net)
    fp.initialize_node_list()
    fp.reset()

    tml_positions = []
    for _ in fp.name2tml.keys():
        tml_positions.append((fp.name2tml[_].x, fp.name2tml[_].y))
    print("total tml num", len(tml_positions))


    total_steps = 0
    temperature = init_temperature
    epoch = 0
    start = time.time()
    previous_save_plt = 0

    best_cost = 1e8
    while True:
        reject = 0
        for _ in range(gap_iter_update_temperature):
            act = np.random.randint(0, 3)
            fp.get_all_nodes_coordinate()
            last_cost = fp.calculate_cost()

            # reshape
            if act == 0:
                blk = np.random.randint(0, fp.num_blk)
                fp.rotate(blk)
                fp.get_all_nodes_coordinate()
                cost = fp.calculate_cost()
                if np.random.rand() < np.exp(-(cost - last_cost) / temperature):
                    pass
                else:
                    reject += 1
                    fp.rotate(blk)
                    
            
            # swap
            elif act == 1:
                b1, b2 = np.random.choice(fp.num_blk, 2, replace=False)
                fp.swap(b1, b2)
                fp.get_all_nodes_coordinate()
                cost = fp.calculate_cost()
                if np.random.rand() < np.exp(-(cost - last_cost) / temperature):
                    pass
                else:
                    reject += 1
                    fp.swap(b1, b2)

            # move
            elif act == 2:
                while True:
                    node_del, node_ins = np.random.choice(fp.num_blk, 2, replace=False)
                    if fp.node_list[node_del].prev is None:
                        continue
                    elif fp.node_list[node_del].left is not None and fp.node_list[node_del].right is not None:
                        continue
                    elif fp.node_list[node_ins].left is not None and fp.node_list[node_ins].right is not None:
                        continue
                    else:
                        break

                t = fp.delandins(node_del, node_ins)
                fp.get_all_nodes_coordinate()
                cost = fp.calculate_cost()
                if np.random.rand() < np.exp(-(cost - last_cost) / temperature):
                    pass
                else:
                    reject += 1
                    fp.revert_delandins(*t)

            # move tree
            elif act == 3:
                while True:
                    node_del, node_ins = np.random.choice(fp.num_blk, 2, replace=False)
                    if fp.node_list[node_del].prev is None:
                        continue
                    elif fp.node_list[node_ins].left is not None and fp.node_list[node_ins].right is not None:
                        continue
                    elif node_ins in fp.get_subtree_node_indices(node_del):
                        continue
                    else:
                        break

                t = fp.move_tree(node_del, node_ins)
                fp.get_all_nodes_coordinate()
                cost = fp.calculate_cost()
                if np.random.rand() < np.exp(-(cost - last_cost) / temperature):
                    pass
                else:
                    reject += 1
                    fp.revert_move_tree(*t)
            total_steps +=1

            if  cost < best_cost:
                best_cost = cost
                fp.write_report(time.time() - s, output)
            if total_steps - previous_save_plt > 1000:
                previous_save_plt = total_steps
                utils.draw(tml_positions, fp.new_chip_w, fp.new_chip_h, our_wandb, total_steps, output,
                           png_path, fp.num_layer)

        # update temperature
        temperature *= 0.6

        # print description
        reject = reject/gap_iter_update_temperature
        epoch += 1
        current_iter = epoch*gap_iter_update_temperature
        description = 'iter = {:6d}, reject = {:.2f} (%)'.format(current_iter, reject*100)
        print(description)

        our_wandb.log({'Temperature':temperature, 'Cost': fp.calculate_cost(), 'HPWL': fp.HPWL(), 'Area': fp.calculate_area(), 'Feedthrough': fp.calculate_feedthrough(), 'steps': epoch*args.gap_iter_update_temperature,  'iter':epoch, 'Time cost': time.time() - start })

        # record for csv
        fp.get_all_nodes_coordinate()
        line = pd.DataFrame([{
            "iter": current_iter,
            "reject": reject,
            "cost": fp.calculate_cost(),
            "hpwl": fp.HPWL(),
            "area": fp.calculate_area(),
            "feedthrough": fp.calculate_feedthrough(),
            "temperature": temperature,
            "time": time.time() - s,
        }])
        df = pd.concat([df, line])
        df.to_csv(csv_path, index=None)


        # draw cost curve
        if enable_draw:
            plt.figure(figsize=(20,10))
            for subplot_idx, col in enumerate(df.columns):
                x = df['iter'].values
                y = df[col].values
                plt.subplot(2,4,subplot_idx+1)
                plt.plot(x,y,label=col)
                plt.legend()
                plt.title(col)
                plt.grid(which='both')
            plt.savefig(os.path.join(result_dir, "curve.png"))
            plt.close()


        # terminate
        if reject > 0.995:
            print('reject limit')
            break


    e = time.time()
    fp.write_report(e-s, output)

    utils.save_json(fp.summary(epoch*gap_iter_update_temperature),os.path.join(result_dir, "summary.json"))





if __name__ == "__main__":
    args = get_args()
    utils.mkdir(args.result_dir, rm=True)
    utils.save_json(vars(args), os.path.join(args.result_dir, "args.json"))

    name = "SA_Env_"+ str(args.circuit) + "_" + str(args.weight_hpwl) + "_"+str(args.weight_area) + "_" + str(args.gap_iter_update_temperature)
    our_wandb = wandb.init(project="EDA_FP", name=name)

    input_blk = "input_pa2/{}.block".format(args.circuit)
    input_net = "input_pa2/{}.nets".format(args.circuit)

    output    = os.path.join(args.result_dir, "circuit={}.txt".format(args.circuit,))
    png_path  = output.replace(".txt", ".png")


    main(input_blk, input_net, output, png_path, args.enable_draw, args.num_layer, args.gap_iter_update_temperature, args.result_dir, args.weight_hpwl, args.weight_area, args.weight_feedthrough, args.init_temperature)

    
    
