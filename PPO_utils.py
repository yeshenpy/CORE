"""
CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
Paper: https://openreview.net/forum?id=86IvZmY26S
Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
         Zhentao Tang, Mingxuan Yuan, Junchi Yan
License: Non-Commercial License (see LICENSE). Commercial use requires permission.
Signature: CORE Authors (NeurIPS 2025)
"""

import torch
import numpy as np
from torch_geometric.data import DataLoader
import torch.nn as nn

import random


def get_init_action(fp, name2index, num_blk):

    current_fp_name2blk = fp.name2blk

    insert_name_list = list(current_fp_name2blk.keys())

    insert_name = random.choice(insert_name_list)
    target_name = "000"
    lrr = 1
    return insert_name, target_name, lrr

def new_mask(name2blk, name2index, num_blk, dict_on, dict_left, dict_right):

    insert_mask = np.zeros([num_blk])

    target_mask= np.zeros([num_blk])

    left_right_rotate_mask = np.zeros([num_blk, 4])

    name_list = name2blk.keys()

    for _ in name_list:
        # 没有在FP上的，才可以被插入

        if dict_on[_] == 0:
            index = name2index[_]
            insert_mask[index] = 1
        else:
            index = name2index[_]
            if dict_left[_] == 0 or dict_right[_] == 0:
                target_mask[index] = 1

            if dict_left[_] == 0:
                left_right_rotate_mask[index, 0] = 1
                left_right_rotate_mask[index, 1] = 1
            if dict_right[_] == 0:
                left_right_rotate_mask[index, 2] = 1
                left_right_rotate_mask[index, 3] = 1
    return insert_mask, target_mask, left_right_rotate_mask


def data_to_PPO_data(args, device, selected_fp, data, name2index, blk_index_to_name, blk_num, tml_index_to_name, tml_num, global_edje_infos, tml_states):
    init_name2blk = selected_fp.name2blk

    selected_fp.reset_wo_init_tree()

    obs_node_info = []
    obs_adj_mat = []
    actions = []
    insert_masks = []
    target_masks = []
    left_right_rotate_masks = []


    insert_first = None
    org_w = None
    org_h = None

    blk_name_left = {}
    blk_name_right = {}
    blk_name_on = {}
    name_list = init_name2blk.keys()
    for name in name_list:
        blk_name_left[name] = 0
        blk_name_right[name] = 0
        blk_name_on[name] = 0


    for step, SA_action in enumerate(data):
        if step == 0:
            if init_name2blk[SA_action[0]].w == SA_action[3] and init_name2blk[SA_action[0]].h == SA_action[4]:
                rotate_left_right = 0
            else:
                rotate_left_right = 1
            org_w = SA_action[3]
            org_h = SA_action[4]
            insert_first = SA_action[0]

            blk_name_on[SA_action[0]] = 1
            selected_fp.Insert_to_target_left_right_rotate(step, SA_action[0], SA_action[1], rotate_left_right)
            selected_fp.get_all_nodes_coordinate()

        else:
            insert_mask, target_mask, left_right_rotate_mask = new_mask(init_name2blk, name2index, blk_num, blk_name_on, blk_name_left, blk_name_right)
            # org_insert_mask, org_target_mask, org_left_right_rotate_mask = all_mask(selected_fp, name2index, blk_num)
            # assert  np.array_equal(insert_mask, org_insert_mask)
            # assert  np.array_equal(target_mask, org_target_mask)
            # assert np.array_equal(left_right_rotate_mask, org_left_right_rotate_mask)

            insert_mask = torch.tensor(insert_mask, dtype=torch.int64).to(device)
            target_mask = torch.tensor(target_mask, dtype=torch.int64).to(device)
            left_right_rotate_mask = torch.tensor(left_right_rotate_mask, dtype=torch.int64).to(device)

            nodeinfos = selected_fp.getBlkInfoMap()
            blk_states = [nodeinfos[blk_index_to_name[index]] for index in range(blk_num)]
            # tml_states = [nodeinfos[tml_index_to_name[index]] for index in
            #               range(blk_num, blk_num + tml_num)]
            graph_states = torch.FloatTensor(np.concatenate([blk_states, tml_states], 0)).to(device)

            #edj_attr = torch.FloatTensor(selected_fp.edge_attr()).to(device)

            # np.transpose(selected_fp.edge_info())

            adj_matrix = torch.tensor(global_edje_infos, dtype=torch.int64).to(device)

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

            #sequence_wh.append([init_name2blk[SA_action[0]].w, SA_action[3]])

            sa_action = torch.LongTensor(
                np.array([name2index[SA_action[0]], name2index[SA_action[1]], rotate_left_right])).to(
                device).unsqueeze(0)

            selected_fp.Insert_to_target_left_right_rotate(step, SA_action[0], SA_action[1],
                                                           rotate_left_right)

            selected_fp.get_all_nodes_coordinate()
            # note negative， lower cost is preferred
            obs_node_info.append(graph_states)
            obs_adj_mat.append(adj_matrix)
            actions.append(sa_action)
            insert_masks.append(insert_mask)
            target_masks.append(target_mask)
            left_right_rotate_masks.append(left_right_rotate_mask)

            blk_name_on[SA_action[0]] = 1


            if rotate_left_right == 0 or rotate_left_right == 1:
                blk_name_left[SA_action[1]] = 1
            else :
                blk_name_right[SA_action[1]] = 1



    return obs_node_info, obs_adj_mat, actions, insert_masks, target_masks, left_right_rotate_masks, insert_first, org_w, org_h



def data_to_PPO_data_feature_version(args, device, selected_fp, data, name2index, blk_index_to_name, blk_num, tml_index_to_name, tml_num):
    init_name2blk = selected_fp.name2blk

    selected_fp.reset_wo_init_tree()


    selected_fp.get_all_nodes_coordinate()

    actions = []
    rewards = []



    sequence = []
    sequence_index = []
    sequence_target = []
    sequence_wh = []

    llr_sequence = []


    state_feature = []

    insert_first = None
    org_w = None
    org_h = None



    for step, SA_action in enumerate(data):
        if step == 0:
            #state_feature.append(torch.FloatTensor(selected_fp.Graph_feature()).to(device))
            if init_name2blk[SA_action[0]].w == SA_action[3] and init_name2blk[SA_action[0]].h == SA_action[4]:
                rotate_left_right = 0
            else:
                rotate_left_right = 1
            sequence.append(SA_action[0])
            sequence_target.append(SA_action[1])
            llr_sequence.append(rotate_left_right)
            sequence_wh.append([init_name2blk[SA_action[0]].w ,SA_action[3]])
            sequence_index.append(name2index[SA_action[0]])

            org_w = SA_action[3]
            org_h = SA_action[4]
            insert_first = SA_action[0]
            selected_fp.Insert_to_target_left_right_rotate(step, SA_action[0], SA_action[1], rotate_left_right)
            selected_fp.get_all_nodes_coordinate()
            area = selected_fp.calculate_area()
            HPWL = selected_fp.HPWL()
            outbound = selected_fp.calculate_outbound()
            reward = -args.A_weight * area - args.L_weight * HPWL - args.O_weight * outbound

        else:

            state_feature.append(torch.FloatTensor(selected_fp.Graph_feature()).to(device))
            #nodeinfos = selected_fp.getNodeInfoMap()
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

            sequence_wh.append([init_name2blk[SA_action[0]].w, SA_action[3]])

            sa_action = torch.LongTensor(
                np.array([name2index[SA_action[0]], name2index[SA_action[1]], rotate_left_right])).to(
                device).unsqueeze(0)
            llr_sequence.append(rotate_left_right)

            selected_fp.Insert_to_target_left_right_rotate(step, SA_action[0], SA_action[1],
                                                           rotate_left_right)
            sequence_index.append(name2index[SA_action[0]])
            sequence.append(SA_action[0])
            sequence_target.append(SA_action[1])
            selected_fp.get_all_nodes_coordinate()
            # note negative， lower cost is preferred
            area = selected_fp.calculate_area()
            HPWL = selected_fp.HPWL()
            outbound = selected_fp.calculate_outbound()
            reward = -args.A_weight * area - args.L_weight * HPWL - args.O_weight * outbound
            actions.append(sa_action)
            rewards.append(torch.tensor(reward - previous_reward).to(device).unsqueeze(0))

        previous_reward = reward
    state_feature.append(torch.FloatTensor(selected_fp.Graph_feature()).to(device))

    return state_feature[0:-1], state_feature[1::], actions, rewards, insert_first, org_w, org_h, sequence, HPWL


import torch.nn.functional as F

import time
def train_PPO(args, gpu_policy, optimizer,data, SA_DATA ,gpu_device ):

    value_loss_list = []
    policy_loss_list = []
    old_approx_kl_list = []
    approx_kl_list = []
    entropy_list = []
    BC_list = [0.0]

    for i in range(5):
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
            _, newlogprob, entropy, newvalue, _ = gpu_policy.forward(
                batch_node_info, batch_edge_index, batch_edge_attr, batch_insert_mask, batch_target_mask,
                batch_left_right_rotate_masks, batch_actions)
            logratio = newlogprob - batch_logprobs

            ratio = logratio.exp()
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                old_approx_kl_list.append(old_approx_kl.detach().cpu())
                approx_kl_list.append(approx_kl.detach().cpu())
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -batch_advantages * ratio
            pg_loss2 = -batch_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)

            if args.clip_vloss:
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


            loss = pg_loss  - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            if len(SA_DATA) > 0:
                sa_loader = DataLoader(SA_DATA, batch_size=args.batch_size, shuffle=True)
                for SA_batch_data in sa_loader:
                    SA_batch_node_info, SA_batch_edge_index, SA_batch_insert_mask, SA_batch_target_mask, SA_batch_left_right_rotate_masks, SA_batch_actions = SA_batch_data.x.detach().to(
                        gpu_device), SA_batch_data.edge_index.detach().to(gpu_device),  SA_batch_data.insert_mask.detach().to(
                        gpu_device), SA_batch_data.target_mask.detach().to(
                        gpu_device), SA_batch_data.left_right_rotate_mask.detach().to(
                        gpu_device), SA_batch_data.action.detach().to(gpu_device)

                    prob_insert, prob_target, prob_rotate = gpu_policy.get_prob(
                        SA_batch_node_info, SA_batch_edge_index, None, SA_batch_insert_mask,
                        SA_batch_target_mask,
                        SA_batch_left_right_rotate_masks, SA_batch_actions)
                    policy_clone_loss = F.cross_entropy(prob_insert, SA_batch_actions[:, 0].long()) + F.cross_entropy(
                        prob_target[0], SA_batch_actions[:, 1].long()) + F.cross_entropy(prob_rotate[0],SA_batch_actions[:, 2].long())

                    loss += args.in_alpha*policy_clone_loss
                    BC_list.append(policy_clone_loss.detach().cpu())
                    break

            value_loss_list.append(v_loss.detach().cpu())
            policy_loss_list.append(pg_loss.detach().cpu())
            entropy_list.append(entropy_loss.detach().cpu())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(gpu_policy.parameters(), 0.5)
            optimizer.step()
        print("train iter", i, " end ....", time.time() - aaaa, loss)

    return np.mean(BC_list),np.mean(value_loss_list), np.mean(policy_loss_list), np.mean(old_approx_kl_list),  np.mean(approx_kl_list), np.mean(entropy_list), np.mean(clipfracs)
