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
import copy
import wandb
import torch

import psutil
from torch_geometric.data import Dataset, Data, DataLoader
import torch.nn.functional as F
from Net import CateoricalPolicy
import torch.optim as optim
import torch.nn as nn

pid = os.getpid()
p = psutil.Process(pid)


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

def get_policy_data_for_PPO(nums, fp, insert, llr,  gpu_policy, name2index, cpu_device, gpu_device, blk_index_to_name, tml_index_to_name, blk_num, tml_num, all_index_to_name):
    data = []
    HPWL_list = []
    for _ in range(nums):
        obs_node_info = []
        obs_adj_mat = []
        obs_att_mat = []
        actions = []
        logprobs = []
        rewards = []
        values = []
        insert_masks = []
        target_masks = []

        total_steps = 0
        left_right_rotate_masks = []
        # fp.get_all_nodes_coordinate()
        fp.reset_wo_init_tree()
        reward_list = []
        with torch.no_grad():
            for step in range(args.one_epsodic_length):
                total_steps += 1
                # print("outer", step)
                if step == 0:
                    target = ""
                    #   print( insert, target, llr)
                    fp.Insert_to_target_left_right_rotate(step, insert, target, llr)
                    fp.get_all_nodes_coordinate()
                    area = fp.calculate_area()
                    HPWL = fp.HPWL()
                    outbound = fp.calculate_outbound()
                    reward = -args.A_weight * area - args.L_weight * HPWL - args.O_weight * outbound
                else:
                    insert_mask, target_mask, left_right_rotate_mask = all_mask(fp, name2index)
                    #  print("1.1")
                    # print("del_mask ", del_mask, "del_mask_left_right ", del_mask_left_right)
                    insert_mask = torch.tensor(insert_mask, dtype=torch.int64).to(cpu_device)
                    target_mask = torch.tensor(target_mask, dtype=torch.int64).to(cpu_device)
                    left_right_rotate_mask = torch.tensor(left_right_rotate_mask, dtype=torch.int64).to(cpu_device)
                    #  print("1.2")

                    nodeinfos = fp.getNodeInfoMap()
                    #  print("1.3")
                    blk_states = [nodeinfos[blk_index_to_name[index]] for index in range(blk_num)]
                    tml_states = [nodeinfos[tml_index_to_name[index]] for index in range(blk_num, blk_num + tml_num)]
                    graph_states = torch.FloatTensor(np.concatenate([blk_states, tml_states], 0)).to(cpu_device)
                    #   print("1.4")
                    edj_attr = torch.FloatTensor(fp.edge_attr()).to(cpu_device)
                    #   print("1.5")
                    adj_matrix = torch.tensor(np.transpose(fp.edge_info()), dtype=torch.int64).to(cpu_device)
                    gpu_policy.eval()
                    action, logp, entropy, value = gpu_policy.sample_action(graph_states.to(gpu_device),
                                                                            adj_matrix.to(gpu_device),
                                                                            edj_attr.to(gpu_device),
                                                                            insert_mask.to(gpu_device),
                                                                            target_mask.to(gpu_device),
                                                                            left_right_rotate_mask.to(gpu_device))
                    # print(step, "aaaa", insert_mask,action)
                    numpy_action = action.detach().cpu().numpy().flatten()
                    # print("i",i, numpy_action)
                    # 操作，得到结果
                    #    print( all_index_to_name[numpy_action[0]], all_index_to_name[numpy_action[1]], numpy_action[2])
                    fp.Insert_to_target_left_right_rotate(step, all_index_to_name[numpy_action[0]],
                                                          all_index_to_name[numpy_action[1]], numpy_action[2])
                    # env_step(fp, numpy_action[0], numpy_action,all_index_to_name)
                    # 此时重新布局，得到reward
                    fp.get_all_nodes_coordinate()
                    # note negative， lower cost is preferred
                    area = fp.calculate_area()
                    HPWL = fp.HPWL()
                    outbound = fp.calculate_outbound()

                    reward = -args.A_weight * area - args.L_weight * HPWL - args.O_weight * outbound
                    reward_list.append(reward - previous_reward)
                    obs_node_info.append(graph_states)
                    obs_adj_mat.append(adj_matrix)
                    # print(step, "??", reward,  reward - previous_reward)
                    obs_att_mat.append(edj_attr)
                    actions.append(action.long())
                    logprobs.append(logp.to(cpu_device))
                    rewards.append(torch.tensor(reward - previous_reward).to(cpu_device).unsqueeze(0))
                    values.append(value[0].to(cpu_device))
                    insert_masks.append(insert_mask)
                    target_masks.append(target_mask)
                    left_right_rotate_masks.append(left_right_rotate_mask)
                previous_reward = reward

        #            print("time cost", time.time() - start)
        print("PPO collect ", _, " HPWL =", HPWL )
        HPWL_list.append(HPWL)
        values = torch.cat(values, 0)
        rewards = torch.cat(rewards, 0)
        with torch.no_grad():
            # nodeinfos = fp.getNodeInfoMap()
            # blk_states = [nodeinfos[blk_index_to_name[index]] for index in range(blk_num)]
            # tml_states = [nodeinfos[tml_index_to_name[index]] for index in range(blk_num, blk_num + tml_num)]
            # graph_states = torch.FloatTensor(np.concatenate([blk_states, tml_states], 0)).to(device)
            # edj_attr = torch.FloatTensor(fp.edge_attr()).to(device)
            # adj_matrix = torch.tensor(np.transpose(fp.edge_info()), dtype=torch.int64).to(device)
            # next_value = policy.get_value(graph_states, adj_matrix, edj_attr).view(-1)
            advantages = torch.zeros_like(rewards).to(cpu_device)
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
        for i in range(args.one_epsodic_length - 1):
            graph_data = Data(value=values[i], logp=logprobs[i], insert_mask=insert_masks[i], target_mask=target_masks[i],
                              left_right_rotate_mask=left_right_rotate_masks[i],
                              x=obs_node_info[i], edge_index=obs_adj_mat[i], edge_attr=obs_att_mat[i], action=actions[i],
                              returns=returns[i], advantage=advantages[i])

            data.append(graph_data)

    return data, np.min(HPWL_list)



def all_mask(fp, name2index):

    insert_mask = np.zeros([fp.num_blk])

    target_mask= np.zeros([fp.num_blk])

    left_right_rotate_mask = np.zeros([fp.num_blk, 4])
    for _ in fp.name2blk.keys():
        # 没有在FP上的，才可以被插入
        if fp.name2blk[_].on_FP == 0:
            index = name2index[_]
            insert_mask[index] = 1
        else:
            index = name2index[_]
            if fp.name2blk[_].left == 0 or fp.name2blk[_].right == 0:
                target_mask[index] = 1

            if fp.name2blk[_].left == 0:
                left_right_rotate_mask[index, 0] = 1
                left_right_rotate_mask[index, 1] = 1
            if fp.name2blk[_].right == 0:
                left_right_rotate_mask[index, 2] = 1
                left_right_rotate_mask[index, 3] = 1
    return insert_mask, target_mask, left_right_rotate_mask

def evaluate_PPO(test_fp, insert_first,rotate_left_right, policy, name2index, blk_index_to_name, tml_index_to_name, blk_num,tml_num, all_index_to_name ):
    # print("org roots", test_fp.roots)
    total_steps = 0
    test_fp.reset_wo_init_tree()
    init_name2blk = test_fp.name2blk
    for step in range(args.one_epsodic_length):
        total_steps += 1
        # print("outer", step)
        if step == 0:

            insert = insert_first
            target = ""


            # print(step, "Insert_name", insert, " ", rotate_left_right)
            test_fp.Insert_to_target_left_right_rotate(step, insert, target, rotate_left_right)
            test_fp.get_all_nodes_coordinate()
            area = test_fp.calculate_area()
            HPWL = test_fp.HPWL()
            outbound = test_fp.calculate_outbound()
            reward = -args.A_weight * area - args.L_weight * HPWL - args.O_weight * outbound

            # print("first name ", insert)
        else:
            insert_mask, target_mask, left_right_rotate_mask = all_mask(test_fp, name2index)
            #  print("1.1")
            # print("del_mask ", del_mask, "del_mask_left_right ", del_mask_left_right)
            insert_mask = torch.tensor(insert_mask, dtype=torch.int64).to(device)
            target_mask = torch.tensor(target_mask, dtype=torch.int64).to(device)
            left_right_rotate_mask = torch.tensor(left_right_rotate_mask, dtype=torch.int64).to(device)
            #  print("1.2")

            nodeinfos = test_fp.getNodeInfoMap()
            #  print("1.3")
            blk_states = [nodeinfos[blk_index_to_name[index]] for index in range(blk_num)]
            tml_states = [nodeinfos[tml_index_to_name[index]] for index in
                          range(blk_num, blk_num + tml_num)]
            graph_states = torch.FloatTensor(np.concatenate([blk_states, tml_states], 0)).to(device)
            #   print("1.4")
            edj_attr = torch.FloatTensor(test_fp.edge_attr()).to(device)
            #   print("1.5")
            adj_matrix = torch.tensor(np.transpose(test_fp.edge_info()), dtype=torch.int64).to(device)
            policy.eval()
            action, logp, entropy, value = policy.sample_action(graph_states, adj_matrix, edj_attr,
                                                                insert_mask, target_mask,
                                                                left_right_rotate_mask)
            # print(step, "aaaa", insert_mask,action)
            numpy_action = action.detach().cpu().numpy().flatten()
            # print("i",i, numpy_action)
            # 操作，得到结果
            #    print( all_index_to_name[numpy_action[0]], all_index_to_name[numpy_action[1]], numpy_action[2])
            test_fp.Insert_to_target_left_right_rotate(step, all_index_to_name[numpy_action[0]],
                                                       all_index_to_name[numpy_action[1]], numpy_action[2])

            # print(step , "Insert_name", all_index_to_name[numpy_action[0]], all_index_to_name[numpy_action[1]], numpy_action[2])
            # env_step(fp, numpy_action[0], numpy_action,all_index_to_name)
            # 此时重新布局，得到reward
            test_fp.get_all_nodes_coordinate()
            # note negative， lower cost is preferred
            area = test_fp.calculate_area()
            HPWL = test_fp.HPWL()
            outbound = test_fp.calculate_outbound()
            reward = -args.A_weight * area - args.L_weight * HPWL - args.O_weight * outbound
    print("after BC the HPWL", HPWL)


def data_to_PPO_data(selected_fp, data, name2index, blk_index_to_name, blk_num, tml_index_to_name, tml_num):
    init_name2blk = selected_fp.name2blk

    selected_fp.reset_wo_init_tree()

    obs_node_info = []
    obs_adj_mat = []
    actions = []
    rewards = []
    insert_masks = []
    target_masks = []
    left_right_rotate_masks = []

    insert_first = None
    first_rotate_left_right = None
    for step, SA_action in enumerate(data):
        if step == 0:
            #   print( insert, target, llr)
            if init_name2blk[SA_action[0]].w == SA_action[3] and init_name2blk[SA_action[0]].h == SA_action[4]:
                rotate_left_right = 0
            else:
                rotate_left_right = 1

            insert_first = SA_action[0]
            first_rotate_left_right = rotate_left_right
            print("Insert_name", SA_action[0], SA_action[1], rotate_left_right)
            selected_fp.Insert_to_target_left_right_rotate(step, SA_action[0], SA_action[1], rotate_left_right)
            selected_fp.get_all_nodes_coordinate()
            area = selected_fp.calculate_area()
            HPWL = selected_fp.HPWL()
            outbound = selected_fp.calculate_outbound()
            reward = -args.A_weight * area - args.L_weight * HPWL - args.O_weight * outbound
        else:
            insert_mask, target_mask, left_right_rotate_mask = all_mask(selected_fp, name2index)
            insert_mask = torch.tensor(insert_mask, dtype=torch.int64).to(device)
            target_mask = torch.tensor(target_mask, dtype=torch.int64).to(device)
            left_right_rotate_mask = torch.tensor(left_right_rotate_mask, dtype=torch.int64).to(device)
            nodeinfos = selected_fp.getNodeInfoMap()
            blk_states = [nodeinfos[blk_index_to_name[index]] for index in range(blk_num)]
            tml_states = [nodeinfos[tml_index_to_name[index]] for index in
                          range(blk_num, blk_num + tml_num)]
            graph_states = torch.FloatTensor(np.concatenate([blk_states, tml_states], 0)).to(device)
            edj_attr = torch.FloatTensor(selected_fp.edge_attr()).to(device)
            adj_matrix = torch.tensor(np.transpose(selected_fp.edge_info()), dtype=torch.int64).to(
                device)
            if SA_action[2] == 1:
                if init_name2blk[SA_action[0]].w == SA_action[3] and init_name2blk[SA_action[0]].h == \
                        SA_action[4]:
                    rotate_left_right = 0
                elif init_name2blk[SA_action[0]].h == SA_action[3] and init_name2blk[SA_action[0]].w == \
                        SA_action[4]:
                    rotate_left_right = 1
                else:
                    raise ValueError(
                        f"Invalid rotate state for block={SA_action[0]} with (w,h)=({SA_action[3]},{SA_action[4]})"
                    )
            else:
                assert SA_action[2] == 2
                if init_name2blk[SA_action[0]].w == SA_action[3] and init_name2blk[SA_action[0]].h == \
                        SA_action[4]:
                    rotate_left_right = 2
                elif init_name2blk[SA_action[0]].h == SA_action[3] and init_name2blk[SA_action[0]].w == \
                        SA_action[4]:
                    rotate_left_right = 3
                else:
                    raise ValueError(
                        f"Invalid rotate state for block={SA_action[0]} with (w,h)=({SA_action[3]},{SA_action[4]})"
                    )
            sa_action = torch.LongTensor(
                np.array([name2index[SA_action[0]], name2index[SA_action[1]], rotate_left_right])).to(
                device).unsqueeze(0)

            selected_fp.Insert_to_target_left_right_rotate(step, SA_action[0], SA_action[1],
                                                           rotate_left_right)

            selected_fp.get_all_nodes_coordinate()
            # note negative， lower cost is preferred
            area = selected_fp.calculate_area()
            HPWL = selected_fp.HPWL()
            outbound = selected_fp.calculate_outbound()
            reward = -args.A_weight * area - args.L_weight * HPWL - args.O_weight * outbound
            obs_node_info.append(graph_states)
            obs_adj_mat.append(adj_matrix)
            actions.append(sa_action)
            rewards.append(torch.tensor(reward - previous_reward).to(device).unsqueeze(0))
            insert_masks.append(insert_mask)
            target_masks.append(target_mask)
            left_right_rotate_masks.append(left_right_rotate_mask)
        previous_reward = reward

    return  obs_node_info, obs_adj_mat, actions, rewards, insert_masks, target_masks, left_right_rotate_masks,insert_first,first_rotate_left_right

def train_PPO(gpu_policy, optimizer,data, gpu_device ):

    value_loss_list = []
    policy_loss_list = []
    old_approx_kl_list = []
    approx_kl_list = []
    entropy_list = []

    for i in range(5):
        print("train iter", i)

        aaaa = time.time()
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
        clipfracs = []
        for batch_data in loader:
            batch_values, batch_logprobs, batch_node_info, batch_edge_index, batch_edge_attr, batch_insert_mask, batch_target_mask, batch_left_right_rotate_masks, batch_returns, batch_actions, batch_advantages = batch_data.value.detach(), batch_data.logp.detach(), batch_data.x.detach(), batch_data.edge_index.detach(), batch_data.edge_attr.detach(), batch_data.insert_mask.detach(), batch_data.target_mask.detach(), batch_data.left_right_rotate_mask.detach(), batch_data.returns.detach(), batch_data.action.detach(), batch_data.advantage.detach()


            if batch_values.shape[0] <= 1:
                continue

            batch_values = batch_values.to(gpu_device)
            batch_logprobs = batch_logprobs.to(gpu_device)
            batch_node_info = batch_node_info.to(gpu_device)
            batch_edge_index = batch_edge_index.to(gpu_device)
            batch_edge_attr = batch_edge_attr.to(gpu_device)
            batch_insert_mask = batch_insert_mask.to(gpu_device)
            batch_target_mask = batch_target_mask.to(gpu_device)
            batch_left_right_rotate_masks = batch_left_right_rotate_masks.to(gpu_device)
            batch_returns = batch_returns.to(gpu_device)
            batch_actions = batch_actions.to(gpu_device)
            batch_advantages = batch_advantages.to(gpu_device)

            # print("1111",torch.cuda.memory_allocated())
            # print("?",batch_values.dtype,batch_logprobs.dtype, batch_node_info.dtype, batch_edge_index.dtype, batch_edge_attr.dtype,  batch_insert_mask.dtype, batch_target_mask.dtype,
            # batch_left_right_rotate_masks.dtype,batch_returns.dtype, batch_actions.dtype, batch_advantages.dtype )
            # print(batch_node_info.dtype,  batch_edge_index.dtype, batch_edge_attr.dtype )
            _, newlogprob, entropy, newvalue, prob_target, prob_insert, prob_rotate = gpu_policy.forward(
                batch_node_info, batch_edge_index, batch_edge_attr, batch_insert_mask, batch_target_mask,
                batch_left_right_rotate_masks, batch_actions)

            #   print("1.5",torch.cuda.memory_allocated())
            logratio = newlogprob - batch_logprobs

            # if iter_epoch == 0 and i == 0:
            #     print("First check", logratio)

            ratio = logratio.exp()
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                old_approx_kl_list.append(old_approx_kl.detach().cpu())
                approx_kl_list.append(approx_kl.detach().cpu())
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            #   print("2222",torch.cuda.memory_allocated())
            # Policy loss
            pg_loss1 = -batch_advantages * ratio
            pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)

            if args.clip_vloss:
                # print("aa",newvalue.shape, batch_returns.shape)
                v_loss_unclipped = (newvalue - batch_returns) ** 2
                v_clipped = batch_values + torch.clamp(
                    newvalue - batch_values,
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - batch_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - batch_returns) ** 2).mean()

            entropy_loss = entropy.mean()


            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            value_loss_list.append(v_loss.detach().cpu())
            policy_loss_list.append(pg_loss.detach().cpu())
            entropy_list.append(entropy_loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gpu_policy.parameters(), 0.5)
            optimizer.step()
        #   print("3333",torch.cuda.memory_allocated())
        # print("loss back ", pg_loss, entropy_loss, v_loss)
        print("train iter", i, " end ....", time.time() - aaaa, loss)

    return np.mean(value_loss_list), np.mean(policy_loss_list), np.mean(old_approx_kl_list),  np.mean(approx_kl_list), np.mean(entropy_list)

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
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--archive_size", type=int, default=4)
    parser.add_argument("--pop_insert_freq", type=int, default=100)
    parser.add_argument("--K_NN", type=int, default=1)
    parser.add_argument("--Select_Top_N", type=int, default=1)
    parser.add_argument("--Elite_size", type=int, default=1)
    parser.add_argument("--ppo_pop_size", type=int, default=1)
    parser.add_argument("--ppo_collect_data_nums", type=int, default=1)
    parser.add_argument("--each_ppo_train_epoch", type=int, default=1)
    parser.add_argument("--one_epsodic_length", type=int, default=1)
    parser.add_argument("--PPO_freq", type=int, default=1)


    parser.add_argument("--O_weight", type=float, default=0.0)
    parser.add_argument("--L_weight", type=float, default=1e-4)
    parser.add_argument("--A_weight", type=float, default=0.0)
    args = parser.parse_args()
    args.result_dir = os.path.join( args.result_dir, args.circuit )
    return args

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
try:
    # Optional dependency; used only for profiling.
    from memory_profiler import profile  # type: ignore
except Exception:  # pragma: no cover
    def profile(func):  # type: ignore
        return func
import random

@profile
def main(input_blk:str, input_net:str, output, png_path, enable_draw:bool, num_layer, gap_iter_update_temperature, result_dir:str, weight_hpwl, weight_area, weight_feedthrough, init_temperature):
    s = time.time()

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
        #fp.reset_wo_init_tree()
        ppo_pop.append(fp)

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
        if k in tmlnames:
            tml_index_to_name[name2index[k]] = k
        all_index_to_name[name2index[k]] = k
    blk_num = ppo_pop[0].num_blk
    tml_num = ppo_pop[0].num_tml
    cpu_device = torch.device("cpu")



    Pop = []
    feature_list = []
    HPWL_list = []
    for _ in range(pop_size):
        Pop.append(tree.Floorplanner(num_layer, weight_hpwl, weight_area, weight_feedthrough,))
        Pop[_].parse_blk(input_blk)
        Pop[_].parse_net(input_net)
        Pop[_].initialize_node_list()
        Pop[_].reset(_)
        feature_list.append(Pop[_].Graph_feature())

        Pop[_].get_all_nodes_coordinate()
        HPWL_list.append(Pop[_].SA_HPWL())

    print("start ", HPWL_list)
    print("feature_list", np.array(feature_list).shape)
    tml_positions = []
    for _ in Pop[0].name2tml.keys():
        tml_positions.append((Pop[0].name2tml[_].x,  Pop[0].name2tml[_].y))
    print("total tml num", len(tml_positions))

    HPWL_list = np.array(HPWL_list)


    best_rank = np.argsort(HPWL_list)



    policy_list = []
    optimizer_list = []
    for _ in range(args.ppo_pop_size):
        gpu_policy = CateoricalPolicy(add_res=True, num_blk=blk_num, num_terminal=tml_num, device=device,use_bn=False).to(device)
        policy_list.append(gpu_policy)
        optimizer_list.append(optim.Adam(gpu_policy.parameters(), lr=2.5e-4, eps=1e-5))

    start = time.time()
    inter_times = 0

    # 初始化archive, 加满buffer
    best_fp_archive = []
    best_HPWL_archive = []
    best_feature_archive = []
    for _ in range(args.archive_size):
        best_fp_archive.append(tree.Floorplanner(num_layer, weight_hpwl, weight_area, weight_feedthrough, ))
        best_fp_archive[_].parse_blk(input_blk)
        best_fp_archive[_].parse_net(input_net)
        best_fp_archive[_].initialize_node_list()
        best_fp_archive[_].reset(_)
        best_fp_archive[_].get_all_nodes_coordinate()
        best_fp_archive[_].initializeFrom(Pop[best_rank[_]])
        best_HPWL_archive.append(HPWL_list[best_rank[_]])
        best_feature_archive.append(feature_list[best_rank[_]])

    best_HPWL = 1e8

    best_RL_HPWL = 1e8
    each_sub = int(pop_size/args.Select_Top_N)


    for i in range(1000000000):

        feature_list = []
        for _ in range(pop_size):
            feature_list.append(Pop[_].Graph_feature())



        # 首先所有个体，从一个起点出发与环境交互

        # 每间隔一定部署，从best archive中sample N个最多样化的个体，初始化种群，重新开始探索

        if  i % args.pop_insert_freq == 0 and i > 0:

            norm_data = np.array(best_feature_archive) / np.linalg.norm(np.array(best_feature_archive), axis=1, keepdims=True)
            similarity_matrix = np.dot(norm_data, norm_data.T)
            #        print("?????" ,similarity_matrix.shape)
            np.fill_diagonal(similarity_matrix, -np.inf)  # 将对角线元素设为负无穷，排除自己与自己的相似度
            #max_similarity_values = np.max(similarity_matrix, axis=1)
            avg_max_similarity = np.mean(np.partition(similarity_matrix, -args.K_NN)[:, -args.K_NN:], axis=1)
            rank = np.argsort(avg_max_similarity)

            # 重新初始化种群
            
            #print("0000 ", len(Pop), len(best_fp_archive), len(rank), each_sub)
            for iii in range(args.Select_Top_N):
                for j in range(iii*each_sub, iii*each_sub+each_sub):
                    Pop[j].initializeFrom(best_fp_archive[rank[iii]])

        else :

            rank = np.argsort(HPWL_list)

            sub_space_norm = int(pop_size/args.Elite_size)

            non_elite_index = list(set(list(range(pop_size))) - set(rank[:args.Elite_size]))

            for iii in range(args.Elite_size):

                sampled_index = random.sample(non_elite_index, sub_space_norm -1)

                non_elite_index = list(set(non_elite_index) - set(sampled_index))

                for j in range(len(sampled_index)):

                    Pop[sampled_index[j]].initializeFrom(Pop[rank[iii]])

        for fp_index, fp in enumerate(Pop):

            inter_times +=1
            act = np.random.randint(0, 3)
            fp.get_all_nodes_coordinate()

            # reshape
            if act == 0:
                blk = np.random.randint(0, fp.num_blk)
                fp.rotate(blk)
                fp.get_all_nodes_coordinate()
                # cost = fp.calculate_cost()
                # if np.random.rand() < np.exp(-(cost - last_cost) / temperature):
                #     pass
                # else:
                #     fp.rotate(blk)
            # swap
            elif act == 1:
                b1, b2 = np.random.choice(fp.num_blk, 2, replace=False)
                fp.swap(b1, b2)
                fp.get_all_nodes_coordinate()
                # cost = fp.calculate_cost()
                # if np.random.rand() < np.exp(-(cost - last_cost) / temperature):
                #     pass
                # else:
                #     fp.swap(b1, b2)
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

                fp.delandins(node_del, node_ins)
                fp.get_all_nodes_coordinate()

            current_HPWL = fp.SA_HPWL()
            # 如果好于平均值，那么就加入

            if current_HPWL < np.mean(best_HPWL_archive):

                norm_data = np.array(best_feature_archive) / np.linalg.norm(np.array(best_feature_archive), axis=1, keepdims=True)
                similarity_matrix = np.dot(norm_data, norm_data.T)
                #        print("?????" ,similarity_matrix.shape)
                np.fill_diagonal(similarity_matrix, -np.inf)  # 将对角线元素设为负无穷，排除自己与自己的相似度
                max_similarity_values = np.max(similarity_matrix, axis=1)

                rank = np.argsort(max_similarity_values)

                replace_index = rank[-1]
                if replace_index == np.argmin(best_HPWL_archive):
                    replace_index = rank[-2]

                best_HPWL_archive[replace_index] = current_HPWL
                best_feature_archive[replace_index] = fp.Graph_feature()
                best_fp_archive[replace_index].initializeFrom(fp)

                # 计算余弦相似度，去掉余弦相似度最高的个体，如果他是性能最好的，则去掉余弦相似度第二的个体

            HPWL_list[fp_index] = current_HPWL


            if current_HPWL < best_HPWL:
                best_HPWL = current_HPWL
                output = os.path.join(args.result_dir, "circuit={}.txt".format(args.circuit, ))
                png_path = output.replace(".txt", ".png")
                fp.write_report(time.time() - s, output)
                utils.draw(tml_positions, copy.deepcopy(Pop[0].new_chip_w), copy.deepcopy(Pop[0].new_chip_h), our_wandb, inter_times, output, png_path, copy.deepcopy(Pop[0].num_layer))

        if i % 5000 == 0:

            norm_data = np.array(best_feature_archive) / np.linalg.norm(np.array(best_feature_archive), axis=1, keepdims=True)

            best_similarity_matrix = np.dot(norm_data, norm_data.T)
            best_mean_diversity = (np.sum(best_similarity_matrix) - args.archive_size) / (
                    args.archive_size * args.archive_size - args.archive_size)

            our_wandb.log({'archive_best': np.min(best_HPWL_archive),'best_HPWL': best_HPWL, 'diversity_best': best_mean_diversity,
                           'steps': inter_times, 'iter': i,
                           'Time cost': time.time() - start})
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            kmeans.fit(np.array(best_feature_archive))
            labels = kmeans.labels_

            output = os.path.join(args.result_dir, "circuit={}_vis.txt".format(args.circuit, ))
            png_path = output.replace(".txt", ".png")
            utils.draw_tsne(np.array(best_feature_archive), 4, labels, png_path, our_wandb, inter_times)



        if i % args.PPO_freq == 0:
            # TODO 如果到了一定步数，那么开始BC到PPO，冷启动，solution转化为序列决策数据

            norm_data = np.array(best_feature_archive) / np.linalg.norm(np.array(best_feature_archive), axis=1,
                                                                        keepdims=True)
            similarity_matrix = np.dot(norm_data, norm_data.T)
            #        print("?????" ,similarity_matrix.shape)
            np.fill_diagonal(similarity_matrix, -np.inf)  # 将对角线元素设为负无穷，排除自己与自己的相似度
            # max_similarity_values = np.max(similarity_matrix, axis=1)
            avg_max_similarity = np.mean(np.partition(similarity_matrix, -args.K_NN)[:, -args.K_NN:], axis=1)
            rank = np.argsort(avg_max_similarity)


            for iii in range(args.ppo_pop_size):

                solution_HPWL = best_fp_archive[rank[iii]].SA_HPWL()

                SA_train_data = []

                ppo_pop[iii].initializeFrom(best_fp_archive[rank[iii]])

#                print("ppo_pop ",ppo_pop, iii)
#                print("????", ppo_pop[iii])

                selected_fp = ppo_pop[iii]
                data = selected_fp.get_place_sequence()

                obs_node_info, obs_adj_mat, actions, rewards, insert_masks, target_masks, left_right_rotate_masks, insert_first,first_rotate_left_right = data_to_PPO_data(selected_fp, data, name2index, blk_index_to_name, blk_num, tml_index_to_name, tml_num)


                print("the ", iii , "-th PPO clone from the fp with HPWL ",  selected_fp.SA_HPWL())
                assert solution_HPWL == selected_fp.SA_HPWL()

                rewards = torch.cat(rewards, 0) + 1
                discounted_reward = torch.zeros_like(rewards)
                running_add = 0

                for t in reversed(range(args.one_epsodic_length - 1)):
                    running_add = running_add * args.gamma + rewards[t]
                    discounted_reward[t] = running_add

                for i in range(args.one_epsodic_length - 1):
                    graph_data = Data(insert_mask=insert_masks[i], target_mask=target_masks[i],left_right_rotate_mask=left_right_rotate_masks[i], x=obs_node_info[i], edge_index=obs_adj_mat[i], action=actions[i],returns=discounted_reward[i])
                    SA_train_data.append(graph_data)

                for ___ in range(100):

                    loader = DataLoader(SA_train_data, batch_size=args.batch_size * 4, shuffle=True)

                    for batch_data in loader:
                        batch_node_info, batch_edge_index, batch_insert_mask, batch_target_mask, batch_left_right_rotate_masks, batch_returns, batch_actions = batch_data.x.detach(), batch_data.edge_index.detach(), batch_data.insert_mask.detach(), batch_data.target_mask.detach(), batch_data.left_right_rotate_mask.detach(), batch_data.returns.detach(), batch_data.action.detach()
                        # print(batch_node_info.dtype,  batch_edge_index.dtype, batch_edge_attr.dtype )
                        _, newlogprob, dist_entropy, newvalue, logits_list = policy_list[iii].forward(batch_node_info, batch_edge_index, None, batch_insert_mask,
                                                                                            batch_target_mask, batch_left_right_rotate_masks, batch_actions)
                        policy_clone_loss = F.cross_entropy(logits_list[0], batch_actions[:, 0].long()) + F.cross_entropy(logits_list[1][0],batch_actions[:,1].long()) + F.cross_entropy( logits_list[2][0], batch_actions[:, 2].long())
                        Critic_clone_loss = torch.mean((newvalue - batch_returns) ** 2)

                        loss = policy_clone_loss + Critic_clone_loss
                        optimizer_list[iii].zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy_list[iii].parameters(), 0.5)
                        optimizer_list[iii].step()
                    print("policy_clone_loss", policy_clone_loss, " Critic_clone_loss", Critic_clone_loss)
                    evaluate_HPWL = evaluate_PPO( ppo_pop[iii], insert_first, first_rotate_left_right, policy_list[iii], name2index, blk_index_to_name, tml_index_to_name, blk_num, tml_num, all_index_to_name )

                    if evaluate_HPWL == solution_HPWL:
                        break
                    else :
                        print("current", evaluate_HPWL, "  target", solution_HPWL)
                # TODO check BC hpwl, 开始训练


                best_foouned_PPO = []

                for ____ in range(args.each_ppo_train_epoch):

                    PPO_data, best_HPWL_founed = get_policy_data_for_PPO(args.ppo_collect_data_nums, policy_list[iii], insert_first, first_rotate_left_right, ppo_pop[iii], name2index, cpu_device, device ,blk_index_to_name, tml_index_to_name, blk_num, tml_num, all_index_to_name)

                    best_foouned_PPO.append(best_HPWL_founed)
                    value_loss, policy_loss, old_approx_kl, approx_kl, entropy_loss = train_PPO(policy_list[iii], optimizer_list[iii], PPO_data, device)


                if np.min(best_foouned_PPO)  < best_RL_HPWL:
                    best_RL_HPWL = np.min(best_foouned_PPO)



                our_wandb.log({'best_RL_HPWL': best_RL_HPWL,
                               'steps': inter_times, 'iter': i,
                               'Time cost': time.time() - start})

                print("After PPO training, best HPWL", np.min(best_foouned_PPO))
if __name__ == "__main__":
    args = get_args()
    pop_size= args.pop_size




    # parser.add_argument("--archive_size", type=int, default=4)
    # parser.add_argument("--pop_insert_freq", type=int, default=100)
    #
    # parser.add_argument("--K_NN", type=int, default=1)
    #
    # parser.add_argument("--Select_Top_N", type=int, default=1)
    #
    # parser.add_argument("--Elite_size", type=int, default=1)
    if args.device == -1:
        device = "cpu"
    else :
        device = "cuda"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    name = "V2_QD_EA_archive_size_"+str(args.archive_size)+"_freq_"+str(args.pop_insert_freq)+ "_knn_" +  str(args.K_NN)+ "_top_n_" + str(args.Select_Top_N)+ "_elite_size_"+ str(args.Elite_size) + "_Env_"+ str(args.circuit) + "_pop_size_" + str(pop_size)  + "_" + str(args.weight_hpwl) + "_"+str(args.weight_area) + "_" + str(args.gap_iter_update_temperature)

    our_wandb = wandb.init(project="EDA_FP", name=name)

    args.result_dir = args.result_dir + "/" + name

    utils.mkdir(args.result_dir, rm=True)
    utils.save_json(vars(args), os.path.join(args.result_dir, "args.json"))

    input_blk = "input_pa2/{}.block".format(args.circuit)
    input_net = "input_pa2/{}.nets".format(args.circuit)

    output    = os.path.join(args.result_dir, "circuit={}.txt".format(args.circuit,))
    png_path  = output.replace(".txt", ".png")


    main(input_blk, input_net, output, png_path, args.enable_draw, args.num_layer, args.gap_iter_update_temperature, args.result_dir, args.weight_hpwl, args.weight_area, args.weight_feedthrough, args.init_temperature)

    
    
