"""
CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
Paper: https://openreview.net/forum?id=86IvZmY26S
Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
         Zhentao Tang, Mingxuan Yuan, Junchi Yan
License: Non-Commercial License (see LICENSE). Commercial use requires permission.
Signature: CORE Authors (NeurIPS 2025)

Note: The main entrypoint is `CORE.py`. In earlier internal versions this file was named `EA.py`.
"""

import os
import torch

import numpy as np

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
# NOTE: Do not hardcode secrets in source code.
# If you want to use Weights & Biases, set `WANDB_API_KEY` in your environment.
os.environ.setdefault("WANDB_MODE", "offline")
torch.set_num_threads(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
np.set_printoptions(threshold=np.inf)
os.environ["JOBLIB_START_METHOD"] = "forkserver"

import tree
import time
import os
import argparse
from tqdm import tqdm
import utils
import pandas as pd
import matplotlib.pyplot as plt
import copy
import wandb
from torch_geometric.data import Dataset, Data, DataLoader

import torch.nn.functional as F
import torch.nn as nn
from PPO_utils import data_to_PPO_data, train_PPO, data_to_PPO_data_feature_version, get_init_action, new_mask
from joblib import parallel_backend




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit", type=str, default="n300")
    parser.add_argument("--enable_draw", type=int, default=1)
    parser.add_argument("--result_dir", type=str, default=f"./result-SA")
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--gap_iter_update_temperature", type=int, default=1)
    parser.add_argument("--weight_hpwl", type=float, default=0.0)
    parser.add_argument("--weight_area", type=float, default=0.5)
    parser.add_argument("--weight_feedthrough", type=float, default=0)
    parser.add_argument("--init_temperature", type=float, default=1e6)
    parser.add_argument("--pop_size", type=int, default=100)
    parser.add_argument("--elite_ratio", type=float, default=0.1)

    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--Area_reward_type", type=int, default=1)

    parser.add_argument("--num_cluster", type=int, default=4)
    parser.add_argument("--one_epsodic_length", type=int, default=50)
    parser.add_argument("--num_best", type=int, default=4)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--O_weight", type=float, default=0.0)
    parser.add_argument("--L_weight", type=float, default=1e-4)
    parser.add_argument("--A_weight", type=float, default=0.0)
    # RL
    parser.add_argument("--ppo_pop_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument('--clip-vloss', action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--PPO_inject_freq", type=int, default=1)
    parser.add_argument("--EA_iter", type=int, default=1)
    parser.add_argument("--total_epoch", type=int, default=10000)
    parser.add_argument("--in_alpha", type=float, default=0.1)


    parser.add_argument("--EA_Area_ratio", type=float, default=0.0)
    args = parser.parse_args()

    args.result_dir = os.path.join(args.result_dir, args.circuit)
    return args


from sklearn.cluster import KMeans

try:
    # Optional dependency; used only for profiling.
    from memory_profiler import profile  # type: ignore
except Exception:  # pragma: no cover
    def profile(func):  # type: ignore
        return func

import random
from Net import CateoricalPolicy
import torch.optim as optim

import multiprocessing as mp

mp.set_start_method('spawn', force=True)

args = get_args()

input_blk = "input_pa2/{}.block".format(args.circuit)
input_net = "input_pa2/{}.nets".format(args.circuit)

def is_pareto_efficient(new_point, frontier):
    """
    Check if a new point is Pareto efficient compared to the current frontier.
    """
    dominated = False
    for point, _ in frontier[:]:
        if all(new_point[i] <= point[i] for i in range(len(new_point))):
            # If new point is worse in all dimensions, it is dominated
            dominated = True
            break
        elif all(new_point[i] >= point[i] for i in range(len(new_point))):
            # If new point is better in all dimensions, it dominates the current point
            frontier.remove((point, _))
    return not dominated

def update_pareto_frontier(new_point, frontier):
    """
    Update Pareto frontier with a new point.
    """
    if is_pareto_efficient(new_point, frontier):
        return True
    return False

def add_point_with_string(new_point, new_string, frontier):
    """
    Add a new point with its associated string to the Pareto frontier if it is efficient.
    """
    if update_pareto_frontier(new_point, frontier):
        frontier.append((new_point, new_string))
        return True
    return False

import sys
class Worker(mp.Process):
    def __init__(self,name_to_size, parameters, experience_queue, index, num_layer, weight_hpwl, weight_area, weight_feedthrough,
                 result_queue, name2index, cpu_device, gpu_device, tml_index_to_name, blk_index_to_name, blk_num,
                 tml_num, all_index_to_name, gpu_policy):
        super(Worker, self).__init__()
        self.p_id = index
        #
        self.seed = index
        self.name_to_size = name_to_size
        self.s = time.time()
        self.parameters = parameters
        self.experience_queue = experience_queue
        self.result_queue = result_queue
        self.name2index = name2index
        self.cpu_device = cpu_device
        self.tml_index_to_name = tml_index_to_name
        self.blk_index_to_name = blk_index_to_name
        self.blk_num = blk_num
        self.tml_num = tml_num
        self.all_index_to_name = all_index_to_name
        self.gpu_policy = gpu_policy
        self.gpu_device = gpu_device

    def get_fp_info(self, fp):
        name2blk_dict  = {}

        name2blk = fp.name2blk
        for key in  name2blk.keys():
            blk = name2blk[key]
            name2blk_dict[key] = [blk.x, blk.y, blk.w, blk.h, blk.order, blk.layer]

        nodeinfo_dict = {}

        node_list = fp.node_list

        for node in node_list:

            if node.prev is None:
                prev_name = ""
            else :
                prev_name = node.prev.name
            if node.right is None:
                right_name = ""
            else :
                right_name = node.right.name
            if node.left is None:
                left_name = ""
            else :
                left_name = node.left.name

            nodeinfo_dict[node.name] = [prev_name, right_name, left_name]

        roots_name = []
        for root in fp.roots:
            roots_name.append(root.name)
        
        fp_info = [name2blk_dict, nodeinfo_dict, roots_name, fp.x_max_each_layer, fp.y_max_each_layer]
        
        return fp_info

    def run(self):

        torch.manual_seed(self.seed)

        fp = tree.Floorplanner(
            args.num_layer,
            args.weight_hpwl,
            args.weight_area,
            args.weight_feedthrough,
        )
        fp.parse_blk(input_blk)
        fp.parse_net(input_net)
        fp.reset_wo_init_tree()


        name2blk = fp.name2blk
        
        while True:
            while self.p_id not in self.parameters:
                time.sleep(0.0001)
            parameters, insert, target, llr  = self.parameters[self.p_id]
            del self.parameters[self.p_id]
            self.gpu_policy.load_state_dict(parameters)

            global_edje_infos = np.transpose(fp.edge_info())
            TML_info = fp.getTmlInfoMap()
            tml_states = [TML_info[self.tml_index_to_name[index]] for index in
                          range(self.blk_num, self.blk_num + self.tml_num)]

            blk_name_left = {}
            blk_name_right = {}
            blk_name_on = {}
            for name in name2blk.keys():
                blk_name_left[name] = 0
                blk_name_right[name] = 0
                blk_name_on[name] = 0


            obs_node_info = []
            obs_adj_mat = []
            obs_att_mat = []
            actions = []
            logprobs = []
            rewards = []
            values = []
            insert_masks = []
            target_masks = []

            left_right_rotate_masks = []
            fp.reset_wo_init_tree()
            reward_list = []

            current_area = 0

            with torch.no_grad():
                for step in range(args.one_epsodic_length):
                    if step == 0:
                        fp.Insert_to_target_left_right_rotate(step, insert, target, llr)
                        fp.get_all_nodes_coordinate()
                        area = fp.calculate_area()
                        HPWL = fp.HPWL()
                        outbound = fp.calculate_outbound()

                        current_area += self.name_to_size[insert]
                        current_area_ratio = current_area / float(area)

                        reward = - args.L_weight * HPWL - args.O_weight * outbound

                        if args.Area_reward_type == 1:
                            reward += args.A_weight * current_area_ratio

                        assert current_area_ratio == 1.0

                        blk_name_on[insert] = 1
                    else:

                        insert_mask, target_mask, left_right_rotate_mask = new_mask(name2blk, self.name2index, self.blk_num, blk_name_on, blk_name_left,blk_name_right)

                        insert_mask = torch.tensor(insert_mask, dtype=torch.int64).to(self.cpu_device)
                        target_mask = torch.tensor(target_mask, dtype=torch.int64).to(self.cpu_device)
                        left_right_rotate_mask = torch.tensor(left_right_rotate_mask, dtype=torch.int64).to(
                            self.cpu_device)

                        nodeinfos = fp.getBlkInfoMap()
                        

                        blk_states = [nodeinfos[self.blk_index_to_name[index]] for index in range(self.blk_num)]
                        
                        graph_states = torch.FloatTensor(np.concatenate([blk_states, tml_states], 0)).to(
                            self.cpu_device)
                        edj_attr = torch.FloatTensor([1.0]).to(self.cpu_device)

                        adj_matrix = torch.tensor(global_edje_infos, dtype=torch.int64).to(self.cpu_device)
                        action, logp, entropy, value = self.gpu_policy.sample_action(graph_states.to(self.cpu_device),
                                                                                     adj_matrix.to(self.cpu_device),
                                                                                     edj_attr.to(self.cpu_device),
                                                                                     insert_mask.to(self.cpu_device),
                                                                                     target_mask.to(self.cpu_device),
                                                                                     left_right_rotate_mask.to(
                                                                                         self.cpu_device))

                        numpy_action = action.detach().cpu().numpy().flatten()
                        fp.Insert_to_target_left_right_rotate(step, self.all_index_to_name[numpy_action[0]],
                                                              self.all_index_to_name[numpy_action[1]], numpy_action[2])
                        fp.get_all_nodes_coordinate()
                        area = fp.calculate_area()
                        HPWL = fp.HPWL()
                        outbound = fp.calculate_outbound()

                        current_area += self.name_to_size[self.all_index_to_name[numpy_action[0]]]
                        current_area_ratio = current_area / float(area)

                        reward = - args.L_weight * HPWL - args.O_weight * outbound

                        if args.Area_reward_type == 1:
                            reward += args.A_weight * current_area_ratio

                        reward_list.append(reward - previous_reward)
                        obs_node_info.append(graph_states.unsqueeze(0))
                        obs_adj_mat.append(adj_matrix.unsqueeze(0))
                        obs_att_mat.append(edj_attr.unsqueeze(0))
                        actions.append(action.unsqueeze(0).long())
                        logprobs.append(logp.to(self.cpu_device).unsqueeze(0))
                        rewards.append(torch.tensor(reward - previous_reward).to(self.cpu_device).unsqueeze(0))
                        values.append(value[0].to(self.cpu_device))
                        insert_masks.append(insert_mask.unsqueeze(0))
                        target_masks.append(target_mask.unsqueeze(0))
                        left_right_rotate_masks.append(left_right_rotate_mask.unsqueeze(0))

                        blk_name_on[self.all_index_to_name[numpy_action[0]]] = 1

                        if numpy_action[2] == 0 or numpy_action[2] == 1:
                            blk_name_left[self.all_index_to_name[numpy_action[1]]] = 1
                        else:
                            blk_name_right[self.all_index_to_name[numpy_action[1]]] = 1


                    previous_reward = reward

            values = torch.cat(values, 0)
            rewards = torch.cat(rewards, 0)
            with torch.no_grad():
                advantages = torch.zeros_like(rewards).to(self.cpu_device)
                lastgaelam = 0
                for t in reversed(range(args.one_epsodic_length - 1)):
                    if t == args.one_epsodic_length - 2:
                        nextnonterminal = 0.0
                        nextvalues = 0.0
                    else:
                        nextnonterminal = 1.0
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

                returns = advantages + values

            logprobs = torch.cat(logprobs, 0).cpu().numpy()
            insert_masks = torch.cat(insert_masks, 0).cpu().numpy()
            target_masks = torch.cat(target_masks, 0).cpu().numpy()
            left_right_rotate_masks = torch.cat(left_right_rotate_masks, 0).cpu().numpy()
            obs_node_info = torch.cat(obs_node_info, 0).cpu().numpy()
            obs_adj_mat = torch.cat(obs_adj_mat, 0).cpu().numpy()
            obs_att_mat = torch.cat(obs_att_mat, 0).cpu().numpy()
            actions = torch.cat(actions, 0).cpu().numpy()

            one_ep_data = (
            values.cpu().numpy(), logprobs, insert_masks, target_masks, left_right_rotate_masks, obs_node_info,
            obs_adj_mat, obs_att_mat, actions, returns.cpu().numpy(), advantages.cpu().numpy())

            fp_info = self.get_fp_info(fp)

            self.experience_queue.put((self.p_id, HPWL, area, outbound, one_ep_data, fp_info, [-HPWL, current_area_ratio]))



def update_worker_networks(gpu_policy, worker_id_list, parameters, fp, name2index,  blk_num):
    gpu_policy.cpu()
    state_dict = gpu_policy.state_dict()
    for p_id in worker_id_list:
        insert, target, llr = get_init_action(fp, name2index, blk_num)
        parameters[p_id] = (state_dict, insert, target, llr)
    gpu_policy.cuda()

@profile
def main(best_output, input_blk: str, input_net: str, output, png_path, enable_draw: bool, num_layer, gap_iter_update_temperature,
         result_dir: str, weight_hpwl, weight_area, weight_feedthrough, init_temperature):
    s = time.time()

    All_fp = []
    name_to_size = {}
    ppo_pop = []
    for _ in range(args.ppo_pop_size):
        fp = tree.Floorplanner(
            num_layer,
            weight_hpwl,
            weight_area,
            weight_feedthrough,
        )
        fp.parse_blk(input_blk)
        fp.parse_net(input_net)
        fp.initialize_node_list()
        fp.reset(_)
        ppo_pop.append(fp)
        All_fp.append(fp)

    data_fp = tree.Floorplanner( num_layer,weight_hpwl, weight_area,weight_feedthrough,)
    data_fp.parse_blk(input_blk)
    data_fp.parse_net(input_net)
    data_fp.initialize_node_list()
    data_fp.reset(100)


    name2index = ppo_pop[0].name2index
    blk_index_to_name = {}
    tml_index_to_name = {}
    all_index_to_name = {}

    name2index = ppo_pop[0].name2index
    blknames = ppo_pop[0].name2blk.keys()
    tmlnames = ppo_pop[0].name2tml.keys()

    for k in name2index.keys():
        if k in blknames:
            blk_index_to_name[name2index[k]] = k
            name_to_size[k] = ppo_pop[0].name2blk[k].w * ppo_pop[0].name2blk[k].h

        if k in tmlnames:
            tml_index_to_name[name2index[k]] = k
        all_index_to_name[name2index[k]] = k
    blk_num = ppo_pop[0].num_blk
    tml_num = ppo_pop[0].num_tml
    cpu_device = torch.device("cpu")

    gpu_policy = CateoricalPolicy(add_res=True, num_blk=blk_num, num_terminal=tml_num, device=device, use_bn=False).to(device)
    gpu_policy.eval()
    optimizer = optim.Adam(gpu_policy.parameters(), lr=2.5e-4, eps=1e-5)

    # store best
    first_blk_to_hpwl = {}
    first_blk_to_data = {}

    tmp_name2blk = ppo_pop[0].name2blk
    for name in blknames:
        blk = tmp_name2blk[name]
        first_blk_to_hpwl[name + "_" + str(blk.w) + "_" + str(blk.h)] = [1e8, 0.0]
        first_blk_to_hpwl[name + "_" + str(blk.h) + "_" + str(blk.w)] = [1e8, 0.0]
    RL_first_blk_to_hpwl = {}



    total_blk_area =  0
    for k in name_to_size.keys():
        total_blk_area +=name_to_size[k]

    print("Total Block Area", total_blk_area)

    Pop = []
    feature_list = []
    HPWL_list = []
    Area_Ratio_list = []
    for _ in range(pop_size):
        Pop.append(tree.Floorplanner(num_layer, weight_hpwl, weight_area, weight_feedthrough, ))
        Pop[_].parse_blk(input_blk)
        Pop[_].parse_net(input_net)
        Pop[_].initialize_node_list()
        Pop[_].reset(_)
        feature_list.append(Pop[_].Graph_feature())

        Pop[_].get_all_nodes_coordinate()
        HPWL_list.append(Pop[_].SA_HPWL())
        Area_Ratio_list.append(-total_blk_area/float(Pop[_].calculate_area()))
        All_fp.append(Pop[_])

    best_fp = tree.Floorplanner(num_layer, weight_hpwl, weight_area, weight_feedthrough, )
    best_fp.parse_blk(input_blk)
    best_fp.parse_net(input_net)
    best_fp.initialize_node_list()
    best_fp.reset(0)

    best_time = None
    print("start ", HPWL_list)
    print("feature_list", np.array(feature_list).shape)
    tml_positions = []
    for _ in Pop[0].name2tml.keys():
        tml_positions.append((Pop[0].name2tml[_].x, Pop[0].name2tml[_].y))
    print("total tml num", len(tml_positions))
    name_to_blk_to_blk_to_index = {}
    for index, key in enumerate(ppo_pop[0].name2blk.keys()):
        name_to_blk_to_blk_to_index[key] = index

    HPWL_list = np.array(HPWL_list)
    Area_Ratio_list = np.array(Area_Ratio_list)
    start = time.time()
    inter_times = 0

    best_HPWL = 1e8
    all_best_HPWL = 1e8
    num_each_class = int(pop_size / args.num_cluster)
    print_steps = 0
    EA_iter =0
    HPWL_best_RL = 1e8
    RL_steps = 0
    Final_All_FP_res = []
    EA_steps = 0
    save_freq = 0


    if not os.path.exists(args.result_dir + "/model"):
        os.makedirs(args.result_dir + "/model")
    torch.save(
        gpu_policy.state_dict(),
        os.path.join(args.result_dir, "model", f"{int(inter_times / 250000)}.pth"),
    )


    result_queue = mp.Queue()
    worker_list =[]

    manager = mp.Manager()
    experience_queue = manager.Queue(args.ppo_pop_size)
    parameters = manager.dict()

    worker_id_list = list(range(args.ppo_pop_size))

    for _ in range(args.ppo_pop_size):
        worder = Worker(name_to_size, parameters, experience_queue, _, num_layer, weight_hpwl, weight_area, weight_feedthrough,
                        result_queue, name2index, cpu_device, device, tml_index_to_name, blk_index_to_name, blk_num,
                        tml_num, all_index_to_name, gpu_policy.cpu())
        worker_list.append(worder)
    for worker in worker_list:
        worker.start()
    update_worker_networks(gpu_policy, worker_id_list, parameters, ppo_pop[0], name2index, blk_num)
    
    
    reset_subprocess_freq = 0
    
    for i in range(args.total_epoch):
        
        
        global_edje_infos = np.transpose(Pop[0].edge_info())

        TML_info = Pop[0].getTmlInfoMap()
        tml_states = [TML_info[tml_index_to_name[index]] for index in range(blk_num, blk_num + tml_num)]

        
        frac = 1.0 - (i - 1.0) / args.total_epoch
        lrnow = frac * 2.5e-4
        
        optimizer.param_groups[0]["lr"] = lrnow

        Final_data = []
        RL_HPWL_list = []
        rank = []
        data = []
        for fp_index in range(args.ppo_pop_size):

            one_data = []

            # this blocks until experience is available
            p_id, HPWL, area, outbound, output_data, fp_info ,  Fp_res = experience_queue.get()
            values, logprobs, insert_masks, target_masks, left_right_rotate_masks, obs_node_info, obs_adj_mat, obs_att_mat, actions, returns, advantages = output_data

            RL_HPWL_list.append(HPWL)
            # string_list.append(output_string)

            print("p id", p_id, HPWL, area, outbound)

            values = torch.FloatTensor(values).to(cpu_device)
            logprobs = torch.FloatTensor(logprobs).to(cpu_device)
            obs_node_info = torch.FloatTensor(obs_node_info).to(cpu_device)
            obs_att_mat = torch.FloatTensor(obs_att_mat).to(cpu_device)
            returns = torch.FloatTensor(returns).to(cpu_device)
            advantages = torch.FloatTensor(advantages).to(cpu_device)
            obs_adj_mat = torch.tensor(obs_adj_mat, dtype=torch.int64).to(cpu_device)
            insert_masks = torch.tensor(insert_masks, dtype=torch.int64).to(cpu_device)
            target_masks = torch.tensor(target_masks, dtype=torch.int64).to(cpu_device)
            left_right_rotate_masks = torch.tensor(left_right_rotate_masks, dtype=torch.int64).to(cpu_device)
            actions = torch.tensor(actions, dtype=torch.int64).to(cpu_device).long()

            for iii in range(args.one_epsodic_length - 1):
                graph_data = Data(value=values[iii], logp=logprobs[iii], insert_mask=insert_masks[iii],
                                  target_mask=target_masks[iii],
                                  left_right_rotate_mask=left_right_rotate_masks[iii],
                                  x=obs_node_info[iii], edge_index=obs_adj_mat[iii], edge_attr=obs_att_mat[iii],
                                  action=actions[iii],
                                  returns=returns[iii], advantage=advantages[iii])
                one_data.append(graph_data)
            data.append(one_data)
            rank.append(p_id)
            inter_times += args.one_epsodic_length
            RL_steps += args.one_epsodic_length
            # Restore floorplanner state from subprocess.
            ppo_pop[fp_index].initializeFrom_info(fp_info[0], fp_info[1], fp_info[2], fp_info[3], fp_info[4])

            our_wandb.log({'RL_HPWL': -Fp_res[0], 'RL_Ratio': Fp_res[1], 'RL_steps': RL_steps})

            if update_pareto_frontier(Fp_res, Final_All_FP_res):
                # ssss = time.time()

                best_time = time.time() - s
                output_string = ppo_pop[fp_index].write_report_string(best_time)

                add_point_with_string(Fp_res, output_string, Final_All_FP_res)

                our_wandb.log({'Parato_HPWL': -Fp_res[0], 'Parato_Ratio': Fp_res[1], 'RL_steps': RL_steps})



            ppo_pop[fp_index].get_all_nodes_coordinate()
            assert  ppo_pop[fp_index].SA_HPWL() == HPWL

            root_name = ppo_pop[fp_index].roots[0].name
            RL_first_blk_to_hpwl[root_name + "_" + str(ppo_pop[fp_index].name2blk[root_name].w) + "_"+ str(ppo_pop[fp_index].name2blk[root_name].h)] = [-Fp_res[0], Fp_res[1]]

            our_wandb.log({'HPWL': HPWL, 'Area': area, 'Out_bound': outbound,
                          'RL_steps':RL_steps ,'steps': inter_times,  'Time cost': time.time() - start})

        rank_list = np.argsort(rank)
        for rrr in rank_list:
            Final_data.extend(data[rrr])
        all_data = Final_data

        if np.min(RL_HPWL_list) < HPWL_best_RL:
            HPWL_best_RL = np.min(RL_HPWL_list)
            # Min_index = np.argmin(HPWL_list)
            if HPWL_best_RL < all_best_HPWL:
                all_best_HPWL = HPWL_best_RL
                best_fp.initializeFrom(ppo_pop[np.argmin(RL_HPWL_list)])
                best_time = time.time() - s
        better_data = []
        for sa_index, name in enumerate(first_blk_to_hpwl.keys()):
            if name in RL_first_blk_to_hpwl and RL_first_blk_to_hpwl[name][0] > first_blk_to_hpwl[name][0] and RL_first_blk_to_hpwl[name][1] < first_blk_to_hpwl[name][1]:
                better_data.extend(first_blk_to_data[name])

        bc_loss, value_loss, policy_loss, old_approx_kl, approx_kl, entropy_loss, clipfracs = train_PPO(args, gpu_policy, optimizer, all_data,better_data,device)
        
        if RL_steps - reset_subprocess_freq > 100000:
            
            reset_s = time.time()
            reset_subprocess_freq =  RL_steps   
            
            
            print("Reset subprocess ...")
            for _, worker in enumerate(worker_list):
                worker.terminate()
                worker.join()
                del worker
                worder = Worker(name_to_size, parameters, experience_queue, _, num_layer, weight_hpwl, weight_area, weight_feedthrough,
                                result_queue, name2index, cpu_device, device, tml_index_to_name, blk_index_to_name, blk_num,
                                tml_num, all_index_to_name, gpu_policy.cpu())
                
                worder.start()
                worker_list[_] = worder
                
            print("All subprocess restart ...", time.time() - reset_s)
        
        update_worker_networks(gpu_policy, worker_id_list, parameters, ppo_pop[0], name2index, blk_num)
        our_wandb.log(
            {'better_data': len(better_data), 'BC_loss': bc_loss, 'HPWL_best_RL': HPWL_best_RL, 'approx_kl_list': np.mean(approx_kl), 'clipfracs': np.mean(clipfracs),
             'old_approx_kl': np.mean(old_approx_kl), 'Entropy': np.mean(entropy_loss), 'PG_loss': np.mean(policy_loss),
             'VF_loss': np.mean(value_loss), 'RL_steps': RL_steps,
             'steps': inter_times,  'Time cost': time.time() - start})
        if inter_times -  save_freq > 250000:
            save_freq = inter_times
            torch.save(
                gpu_policy.state_dict(),
                os.path.join(args.result_dir, "model", f"{int(inter_times / 250000)}.pth"),
            )



        if inter_times - print_steps > 200000:
            print_steps = inter_times

            for siiii, solution in enumerate(Final_All_FP_res):
                utils.draw_string(tml_positions, ppo_pop[0].new_chip_w, ppo_pop[0].new_chip_h, our_wandb, siiii, inter_times, solution[1],
                           png_path, ppo_pop[0].num_layer)


        if  i % args.PPO_inject_freq == 0 and i > 0:
            temp_feature_list = []
            temp_HPWL_list = []
            temp_Area_ratio = []

            for _ in range(len(All_fp)):
                temp_feature_list.append(All_fp[_].Graph_feature())
                temp_HPWL_list.append(All_fp[_].SA_HPWL())
                temp_Area_ratio.append(-total_blk_area/float(All_fp[_].calculate_area()))
            norm_data = np.array(temp_feature_list) / np.linalg.norm(np.array(temp_feature_list), axis=1, keepdims=True)
            similarity_matrix = np.dot(norm_data, norm_data.T)
            np.fill_diagonal(similarity_matrix, -np.inf)
            avg_max_similarity = np.mean(np.partition(similarity_matrix, -2)[:, -2:], axis=1)
            normalized_HPWL = (np.array(temp_HPWL_list) - np.min(temp_HPWL_list)) / (np.max(temp_HPWL_list) - np.min(temp_HPWL_list) + 1e-20)

            normalized_Area_ratio = (np.array(temp_Area_ratio) - np.min(temp_Area_ratio)) / (np.max(temp_Area_ratio) - np.min(temp_Area_ratio) + 1e-20)

            new_fitness = args.alpha * (args.EA_Area_ratio * normalized_Area_ratio + (1 - args.EA_Area_ratio) * normalized_HPWL) + (1 - args.alpha) * avg_max_similarity


            rank_index = np.argsort(new_fitness)

            PPO_better = 0
            for _ in range(pop_size):
               if rank_index[_] < args.ppo_pop_size:
                   PPO_better +=1
               if _ != rank_index[_]:
                   Pop[_].initializeFrom(All_fp[rank_index[_]])
                   HPWL_list[_] = copy.deepcopy(temp_HPWL_list[rank_index[_]])
                   Area_Ratio_list[_] = copy.deepcopy(temp_Area_ratio[rank_index[_]])

            our_wandb.log({'PPO_ratio_in_Pop': PPO_better/len(Pop), 'steps': inter_times, 'iter': i,
                          'Time cost': time.time() - start})

        temp_maping = {}
        for ea_inter in range(args.EA_iter):

            EA_iter +=1
            feature_list = []
            for _ in range(pop_size):
                feature_list.append(Pop[_].Graph_feature())
            norm_data = np.array(feature_list) / np.linalg.norm(np.array(feature_list), axis=1, keepdims=True)
            similarity_matrix = np.dot(norm_data, norm_data.T)

            np.fill_diagonal(similarity_matrix, -np.inf)
            max_similarity_values = np.max(similarity_matrix, axis=1)
            avg_max_similarity = np.mean(np.partition(similarity_matrix, -2)[:, -2:], axis=1)
            normalized_HPWL = (np.array(HPWL_list) - np.min(HPWL_list)) / (np.max(HPWL_list) - np.min(HPWL_list) + 1e-20)

            normalized_Area_ratio = (np.array(Area_Ratio_list) - np.min(Area_Ratio_list)) / (np.max(Area_Ratio_list) - np.min(Area_Ratio_list) + 1e-20)
            new_fitness = args.alpha * (args.EA_Area_ratio* normalized_Area_ratio + (1-args.EA_Area_ratio)*normalized_HPWL) + (1 - args.alpha) * avg_max_similarity

            kmeans = KMeans(n_clusters=args.num_cluster, random_state=42, n_init=10)


            kmeans.fit(feature_list)
            labels = kmeans.labels_

            if EA_iter % 1000 == 0 :
                output = os.path.join(args.result_dir, "circuit={}_vis.txt".format(args.circuit, ))
                png_path = output.replace(".txt", ".png")
                utils.draw_tsne(feature_list, args.num_cluster, labels, png_path, our_wandb, inter_times)

                classes_with_best_feacture = []
                for label in range(args.num_cluster):
                    indices_in_label = np.where(labels == label)[0]
                    sorted_indices = np.argsort(new_fitness[indices_in_label])
                    top_n_indices = indices_in_label[sorted_indices[: 1]]
                    classes_with_best_feacture.append(feature_list[top_n_indices[0]])
                norm_data = np.array(classes_with_best_feacture) / np.linalg.norm(np.array(classes_with_best_feacture), axis=1, keepdims=True)
                best_similarity_matrix = np.dot(norm_data, norm_data.T)
                best_mean_diversity = (np.sum(best_similarity_matrix) - args.num_cluster) / (
                        args.num_cluster * args.num_cluster - args.num_cluster)
                our_wandb.log({'all_best_HPWL':all_best_HPWL, 'best_HPWL': best_HPWL, 'diversity_best': best_mean_diversity,
                               'diversity_all': np.mean(max_similarity_values), 'steps': inter_times, 'iter': i, 'EA_steps':EA_steps,
                               'Time cost': time.time() - start})

            n_top_indices = []

            all_elite_index = []
            for label in range(args.num_cluster):
                indices_in_label = np.where(labels == label)[0]
                sorted_indices = np.argsort(new_fitness[indices_in_label])
                top_n_indices = indices_in_label[sorted_indices[: args.num_best]]
                for elite_index in top_n_indices:
                    all_elite_index.append(elite_index)
            non_elite_index = list(set(list(range(pop_size))) - set(all_elite_index))
            for label in range(args.num_cluster):
                indices_in_label = np.where(labels == label)[0]
                sorted_indices = np.argsort(new_fitness[indices_in_label])
                top_n_indices = indices_in_label[sorted_indices[: args.num_best]]
                n_top_indices.append(top_n_indices)

                class_new_hpwl = []
                class_new_area_ratio = []
                for index in top_n_indices:
                    class_new_hpwl.append(HPWL_list[index])
                    class_new_area_ratio.append(-Area_Ratio_list[index])

                sampled_index = random.sample(non_elite_index, num_each_class - len(top_n_indices))

                non_elite_index = list(set(non_elite_index) - set(sampled_index))

                class_new_pop = []
                for _ in range(num_each_class - len(top_n_indices)):
                    clone_index = np.random.choice(top_n_indices)
                    Pop[sampled_index[_]].initializeFrom(Pop[clone_index])
                    Pop[sampled_index[_]].get_all_nodes_coordinate()
                    assert Pop[sampled_index[_]].SA_HPWL() == Pop[clone_index].SA_HPWL()
                    class_new_pop.append(Pop[sampled_index[_]])

                if EA_iter % 1000 == 0:
                    our_wandb.log({'class_' + str(label)+'_ratio': np.max(class_new_area_ratio),'class_' + str(label): np.min(class_new_hpwl), 'EA_steps':EA_steps, 'steps': inter_times, 'iter': i,
                                   'Time cost': time.time() - start})

                if EA_iter % 1000 == 0:
                    output = os.path.join(args.result_dir, "circuit={}_" + str(label) + ".txt".format(args.circuit, ))
                    png_path = output.replace(".txt", ".png")
                    Pop[top_n_indices[0]].write_report(time.time() - s, output)
                    utils.draw_class(label, tml_positions, copy.deepcopy(Pop[0].new_chip_w),
                                     copy.deepcopy(Pop[0].new_chip_h), our_wandb, inter_times, output, png_path,
                                     copy.deepcopy(Pop[0].num_layer))

                for fp_index, fp in enumerate(class_new_pop):

                    real_fp_index = sampled_index[fp_index]

                    for _ in range(gap_iter_update_temperature):
                        EA_steps +=1
                        inter_times += 1
                        act = np.random.randint(0, 3)
                        fp.get_all_nodes_coordinate()

                        # reshape
                        if act == 0:
                            blk = np.random.randint(0, fp.num_blk)
                            fp.rotate(blk)
                            fp.get_all_nodes_coordinate()

                        # swap
                        elif act == 1:
                            b1, b2 = np.random.choice(fp.num_blk, 2, replace=False)
                            fp.swap(b1, b2)
                            fp.get_all_nodes_coordinate()

                        # move
                        elif act == 2:
                            while True:
                                node_del, node_ins = np.random.choice(blk_num, 2, replace=False)
                                del_node = fp.node_list[node_del]
                                ins_node = fp.node_list[node_ins]
                                if del_node.prev is None:
                                    continue
                                elif del_node.left is not None and del_node.right is not None:
                                    continue
                                elif ins_node.left is not None and ins_node.right is not None:
                                    continue
                                else:
                                    break

                            t = fp.delandins(node_del, node_ins)
                            fp.get_all_nodes_coordinate()

                    current_HPWL = fp.SA_HPWL()
                    current_area_ratio = total_blk_area/float(fp.calculate_area())
                    HPWL_list[real_fp_index] = current_HPWL
                    Area_Ratio_list[real_fp_index] = -current_area_ratio
                    root_name = fp.roots[0].name
                    root_name_w_h = root_name + "_" + str(fp.name2blk[root_name].w) + "_" + str(fp.name2blk[root_name].h)


                    if first_blk_to_hpwl[root_name_w_h][0] > current_HPWL and first_blk_to_hpwl[root_name_w_h][1] < current_area_ratio:


                        
                        if args.in_alpha > 0.0:
                            first_blk_to_hpwl[root_name_w_h] = [current_HPWL, current_area_ratio]
                            ppo_pop[0].initializeFrom(fp)
                            data = ppo_pop[0].get_place_sequence()
                            temp_maping[root_name_w_h] = data

                    if update_pareto_frontier([-current_HPWL, current_area_ratio], Final_All_FP_res):
                        best_time = time.time() - s
                        output_string = fp.write_report_string(best_time)
                        add_point_with_string([-current_HPWL, current_area_ratio], output_string, Final_All_FP_res)

                        our_wandb.log( {'EA_Parato_HPWL': current_HPWL, 'EA_Parato_Ratio': current_area_ratio, 'steps': inter_times})

                    if current_HPWL < best_HPWL:
                        best_HPWL = current_HPWL
                        if best_HPWL < all_best_HPWL:
                            all_best_HPWL = best_HPWL
                            best_fp.initializeFrom(fp)
                            best_time = time.time()- s


            if  EA_iter % 1000 == 0 :
                print(best_fp.write_report_string(best_time))



        if  args.in_alpha > 0.0:
            for map_name in temp_maping.keys():
                obs_node_info, obs_adj_mat, actions, insert_masks, target_masks, left_right_rotate_masks, insert_first, org_w, org_h = data_to_PPO_data(
                    args, cpu_device, ppo_pop[0], temp_maping[map_name], name2index, blk_index_to_name, blk_num, tml_index_to_name, tml_num, copy.deepcopy(global_edje_infos), copy.deepcopy(tml_states))
                data = []
                for iii in range(len(actions)):
                    graph_data = Data(insert_mask=insert_masks[iii],
                                      target_mask=target_masks[iii],
                                      left_right_rotate_mask=left_right_rotate_masks[iii],
                                      x=obs_node_info[iii], edge_index=obs_adj_mat[iii],
                                      action=actions[iii])
                    data.append(graph_data)
                first_blk_to_data[map_name] = data



if __name__ == "__main__":
    pop_size = args.pop_size
    if args.device == -1:
        device = "cpu"
    else:
        device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)




    name = "Parallel_Area_DAPG__RL_"+ str(args.Area_reward_type)+"_EA_" + str(args.EA_Area_ratio) + str(args.A_weight) +  "_" + str(args.total_epoch)+"_ERL_alpha_"+ str(args.in_alpha) +"_inters_"+ str(args.EA_iter) +"_inject_iters_"+str(args.PPO_inject_freq) +"_alpha_" + str(args.alpha) + "_cluster_" + str(args.num_cluster) + "_" + str(
        args.num_best) + "_Env_" + str(args.circuit) + "_pop_size_" + str(pop_size) + "_" + str(args.ppo_pop_size)

    our_wandb = wandb.init(project="EDA_FP", name=name)

    args.result_dir = args.result_dir + "/" + name + "_" + str(args.seed)

    utils.mkdir(args.result_dir, rm=True)
    utils.save_json(vars(args), os.path.join(args.result_dir, "args.json"))

    input_blk = "input_pa2/{}.block".format(args.circuit)
    input_net = "input_pa2/{}.nets".format(args.circuit)

    best_output = os.path.join(args.result_dir, "circuit={}.txt".format(args.circuit, ))

    output = os.path.join(args.result_dir, "circuit={}.txt".format(args.circuit, ))
    png_path = output.replace(".txt", ".png")

    main(best_output, input_blk, input_net, output, png_path, args.enable_draw, args.num_layer, args.gap_iter_update_temperature,
         args.result_dir, args.weight_hpwl, args.weight_area, args.weight_feedthrough, args.init_temperature)
