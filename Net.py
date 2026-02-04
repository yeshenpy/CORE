"""
CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
Paper: https://openreview.net/forum?id=86IvZmY26S
Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
         Zhentao Tang, Mingxuan Yuan, Junchi Yan
License: Non-Commercial License (see LICENSE). Commercial use requires permission.
Signature: CORE Authors (NeurIPS 2025)
"""

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

import torch_geometric.nn as gnn
from torch_geometric.nn import SAGEConv


class GraphSAGE_NET(torch.nn.Module):

    def __init__(self, feature, hidden):
        super(GraphSAGE_NET, self).__init__()
        self.sage1 = SAGEConv(feature, hidden)
        self.sage2 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        return x

# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=2, feats=10,  units=32, act_fn='silu', agg_fn='max', use_bn=False):
        super().__init__()
        self.depth = depth
        self.feats = feats

        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)

        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(2, self.units)

        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.use_bn = use_bn
    def reset_parameters(self):
        raise NotImplementedError

    # [batch_size, 300, 6], [batch_size, 156, 2]
    def forward(self, x, edge_index, edge_attr):
        size = x.shape[0]
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)

        for i in range(self.depth):
            x0 = x

            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            if self.use_bn:
                x = x0 + self.act_fn(x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0], size))
                w = w0 + self.act_fn(w1 + x3[edge_index[0]] + x4[edge_index[1]])
            else:
                x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0], size)))
                w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return x, w


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad=False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.softmax(x, dim=-1)  # last layer
        return x



# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=3, act_fn='silu'):
        self.units = units
        self.preds = preds

        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim=-1)

class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], device="cpu"):
        self.masks = masks
        self.device = device
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)

class TransformerEncoderModel(nn.Module):
    def __init__(self, feat_dim, hidden_size, num_layers, num_heads):
        super(TransformerEncoderModel, self).__init__()

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=feat_dim,
                nhead=num_heads,
                dim_feedforward=hidden_size
            ),
            num_layers=num_layers
        )

    def forward(self, src):
        src = src.permute(1, 0, 2)
        output = self.encoder(src)
        output = output.permute(1, 0, 2)

        return output


class ValueFunction(BaseNetwork):
    def __init__(self, num_blk,num_terminal, units=32, depth=3, device='cpu'):
        super().__init__()
        self.device = device
        self.num_tml = num_terminal
        self.num_blk = num_blk

        self.emb_net = GraphSAGE_NET(10, 64)

        self.units = units
        self.depth = depth
        self.units_list = [self.units] * self.depth
        # init critic
        self.critic = nn.Sequential(*[nn.Linear(64+10, 64), nn.SiLU(), nn.Linear(64, 1)])

        self.transfermer = TransformerEncoderModel(64 + 10, 64, 2, 2)

    def forward(self,x,  edge_index, edge_attr):
        graph_info, node_emb = self.get_node_embeddings(x, edge_index, edge_attr)
        return self.critic(graph_info)

    def get_node_embeddings(self,x,  edge_index, edge_attr):
        node_emb = self.emb_net(x, edge_index)
        node_emb = node_emb.reshape([int(x.shape[0]/(self.num_blk + self.num_tml)), -1, 64])

        node_emb = torch.cat([node_emb, x.reshape([int(x.shape[0]/(self.num_blk + self.num_tml)), -1, 10])], -1)

        node_emb = self.transfermer.forward(node_emb)

        graph_info = node_emb.mean(1)  # [bs, 32]
        return graph_info, node_emb

class CateoricalPolicy(BaseNetwork):

    def __init__(self, add_res, num_blk,num_terminal, units=32, depth=3, device='cpu',use_bn=False):
        super().__init__()
        self.device = device
        self.num_tml = num_terminal
        self.num_blk = num_blk
        self.emb_net = GraphSAGE_NET(10, 64)

        self.units = units
        self.depth = depth
        self.units_list = [self.units] * self.depth

        self.add_res = add_res
        if add_res:
            node_dim = 64 + 10
        else :
            node_dim = 64

        self.transfermer = TransformerEncoderModel(node_dim, node_dim, 2, 2)


        # operator 0
        self.select_insert_score = nn.Sequential(
            *[nn.Linear(node_dim*2 + 10, 64), nn.SiLU(),  nn.Linear(64, 1)])
        # operator 1
        self.select_target_score = nn.Sequential(
            *[nn.Linear(node_dim*3 + 10*2, 64), nn.SiLU(), nn.Linear(64, 1)])
        self.select_left_right_score = nn.Sequential(
            *[nn.Linear(node_dim*3 + 10*2, 64), nn.SiLU(),  nn.Linear(64, 4)])
        # operator 2
        self.critic = ValueFunction(num_blk,num_terminal, units, depth, device)
        # init critic

    def get_value(self,x,  edge_index, edge_attr):
        return self.critic(x,  edge_index, edge_attr)

    def get_insert_blk_logits(self, graph_info, node_emb, org_blk_info):
        logits = self.select_insert_score(torch.cat([graph_info, node_emb, org_blk_info],-1))  # 需要计算出来具体的概率
        return logits.squeeze()

    def get_target_blk_logits(self, graph_info, node_emb, org_blk_info , insert_blk_embedding, org_insert_blk_info):
        logits = self.select_target_score(torch.cat([graph_info, node_emb, org_blk_info, insert_blk_embedding, org_insert_blk_info],-1))  # 需要计算出来具体的概率
        return logits.squeeze()

    def get_left_right_rotate_logits(self, graph_info, insert_blk_embedding,org_insert_blk_info, target_blk_embedding, org_target_blk_info):
        logits = self.select_left_right_score(torch.cat([graph_info, insert_blk_embedding, org_insert_blk_info, target_blk_embedding, org_target_blk_info],-1))  # 需要计算出来具体的概率
        return  logits.squeeze()


    def get_node_embeddings(self,x,  edge_index, edge_attr):
        node_emb = self.emb_net(x, edge_index)
        node_emb = node_emb.reshape([int(x.shape[0]/(self.num_blk + self.num_tml)), -1, 64])
        if self.add_res:
            node_emb = torch.cat([x.reshape(node_emb.shape[0], -1, 10), node_emb], -1)

        node_emb = self.transfermer.forward(node_emb)

        graph_info = node_emb.mean(1)  # [bs, 32]

        return graph_info, node_emb


    def sample_action(self, x,  edge_index, edge_attr, insert_mask, target_mask, left_right_mask):
        batch_size = int(x.shape[0] / (self.num_blk + self.num_tml))

        insert_mask = insert_mask.reshape([batch_size, self.num_blk])
        target_mask = target_mask.reshape([batch_size, self.num_blk])
        left_right_mask = left_right_mask.reshape([batch_size, self.num_blk, -1])

        org_graph_info, node_emb = self.get_node_embeddings(x, edge_index, edge_attr)

        org_blk_info = x.reshape([batch_size,-1, 10])

        graph_info = org_graph_info.unsqueeze(1).expand(-1, self.num_blk, -1)

        insert_blk_logits = self.get_insert_blk_logits(graph_info, node_emb[:, :self.num_blk, :], org_blk_info[:, : self.num_blk,:])

        insert_blk_operator = CategoricalMasked(logits=insert_blk_logits, masks=insert_mask, device="cpu")

        insert_blk_actions = insert_blk_operator.sample()
        insert_entropy = insert_blk_operator.entropy()

        insert_blk_prob = insert_blk_operator.log_prob(insert_blk_actions)

        one_selected_blk_emb = node_emb[torch.arange(node_emb.shape[0]), insert_blk_actions]
        selected_blk_emb = one_selected_blk_emb.unsqueeze(1).expand(-1, self.num_blk, -1)

        one_org_insert_blk_info = org_blk_info[torch.arange(node_emb.shape[0]), insert_blk_actions]
        org_insert_blk_info = one_org_insert_blk_info.unsqueeze(1).expand(-1, self.num_blk, -1)

        target_logits = self.get_target_blk_logits(graph_info, node_emb[:, :self.num_blk, :], org_blk_info[:, : self.num_blk,:] ,selected_blk_emb, org_insert_blk_info)
        target_logits = target_logits.unsqueeze(0)

        target_operator =  CategoricalMasked(logits=target_logits, masks=target_mask, device="cpu")
        target_blk_actions = target_operator.sample()

        target_blk_prob = target_operator.log_prob(target_blk_actions)
        target_entropy = target_operator.entropy()

        one_target_blk_emb = node_emb[torch.arange(node_emb.shape[0]), target_blk_actions]

        one_target_blk_info = org_blk_info[torch.arange(node_emb.shape[0]), target_blk_actions]

        left_right_mask = left_right_mask[torch.arange(left_right_mask.shape[0]), target_blk_actions]

        left_right_rotate_logits = self.get_left_right_rotate_logits(org_graph_info, one_selected_blk_emb,one_org_insert_blk_info, one_target_blk_emb, one_target_blk_info)
        left_right_rotate_logits = left_right_rotate_logits.unsqueeze(0)
        left_right_rotate_operator = CategoricalMasked(logits=left_right_rotate_logits, masks=left_right_mask, device="cpu")
        left_right_rotate_action = left_right_rotate_operator.sample()
        left_right_rotate_logp = left_right_rotate_operator.log_prob(left_right_rotate_action)
        left_right_rotate_entropy = left_right_rotate_operator.entropy()

        logp_sum = left_right_rotate_logp + target_blk_prob + insert_blk_prob
        entropy_sum = left_right_rotate_entropy + target_entropy + insert_entropy

        final_actions = torch.cat([insert_blk_actions.unsqueeze(-1), target_blk_actions.unsqueeze(-1), left_right_rotate_action.unsqueeze(-1)], -1)

        return final_actions, logp_sum, entropy_sum, self.critic(x,  edge_index, edge_attr)

    def greedy_action(self, x, edge_index, edge_attr, insert_mask, target_mask, left_right_mask):
        batch_size = int(x.shape[0] / (self.num_blk + self.num_tml))

        insert_mask = insert_mask.reshape([batch_size, self.num_blk])
        target_mask = target_mask.reshape([batch_size, self.num_blk])
        left_right_mask = left_right_mask.reshape([batch_size, self.num_blk, -1])

        org_graph_info, node_emb = self.get_node_embeddings(x, edge_index, edge_attr)

        org_blk_info = x.reshape([batch_size, -1, 10])

        graph_info = org_graph_info.unsqueeze(1).expand(-1, self.num_blk, -1)




        insert_blk_logits = self.get_insert_blk_logits(graph_info, node_emb[:, :self.num_blk, :],
                                                       org_blk_info[:, : self.num_blk, :])
        insert_mask = insert_mask.type(torch.BoolTensor).to(self.device)
        logits = torch.where(insert_mask, insert_blk_logits, torch.tensor(-1e8).to(self.device))
        insert_blk_actions = torch.argmax(F.softmax(logits, dim=-1))

        one_selected_blk_emb = node_emb[torch.arange(node_emb.shape[0]), insert_blk_actions]
        selected_blk_emb = one_selected_blk_emb.unsqueeze(1).expand(-1, self.num_blk, -1)

        one_org_insert_blk_info = org_blk_info[torch.arange(node_emb.shape[0]), insert_blk_actions]
        org_insert_blk_info = one_org_insert_blk_info.unsqueeze(1).expand(-1, self.num_blk, -1)

        target_logits = self.get_target_blk_logits(graph_info, node_emb[:, :self.num_blk, :],
                                                   org_blk_info[:, : self.num_blk, :], selected_blk_emb,
                                                   org_insert_blk_info)
        target_logits = target_logits.unsqueeze(0)

        target_mask = target_mask.type(torch.BoolTensor).to(self.device)
        logits = torch.where(target_mask, target_logits, torch.tensor(-1e8).to(self.device))
        target_blk_actions = torch.argmax(F.softmax(logits, dim=-1))

        one_target_blk_emb = node_emb[torch.arange(node_emb.shape[0]), target_blk_actions]

        one_target_blk_info = org_blk_info[torch.arange(node_emb.shape[0]), target_blk_actions]

        left_right_mask = left_right_mask[torch.arange(left_right_mask.shape[0]), target_blk_actions]

        left_right_rotate_logits = self.get_left_right_rotate_logits(org_graph_info, one_selected_blk_emb,
                                                                     one_org_insert_blk_info, one_target_blk_emb,
                                                                     one_target_blk_info)
        left_right_rotate_logits = left_right_rotate_logits.unsqueeze(0)

        left_right_mask = left_right_mask.type(torch.BoolTensor).to(self.device)
        logits = torch.where(left_right_mask, left_right_rotate_logits, torch.tensor(-1e8).to(self.device))
        left_right_rotate_action = torch.argmax(F.softmax(logits, dim=-1))

        final_actions = torch.cat([insert_blk_actions.unsqueeze(-1), target_blk_actions.unsqueeze(-1),
                                   left_right_rotate_action.unsqueeze(-1)], -1)

        mask = insert_mask.type(torch.BoolTensor).to(self.device)
        insert_blk_logits = torch.where(mask, insert_blk_logits, torch.tensor(-1e8).to(self.device))

        return final_actions, F.softmax(insert_blk_logits, dim=-1)


    def forward(self, x, edge_index, edge_attr, insert_mask, target_mask, left_right_mask, actions=None):

        batch_size = int(x.shape[0] / (self.num_blk + self.num_tml))

        insert_mask = insert_mask.reshape([batch_size, self.num_blk])
        target_mask = target_mask.reshape([batch_size, self.num_blk])
        left_right_mask = left_right_mask.reshape([batch_size, self.num_blk, -1])

        org_graph_info, node_emb = self.get_node_embeddings(x, edge_index, edge_attr)

        org_blk_info = x.reshape([batch_size, -1, 10])

        graph_info = org_graph_info.unsqueeze(1).expand(-1, self.num_blk, -1)

        insert_blk_logits = self.get_insert_blk_logits(graph_info, node_emb[:, :self.num_blk, :],
                                                       org_blk_info[:, : self.num_blk, :])

        insert_blk_operator = CategoricalMasked(logits=insert_blk_logits, masks=insert_mask, device=self.device)
        if actions is not None:
            insert_blk_actions = actions[:, 0]
        else:
            insert_blk_actions = insert_blk_operator.sample()
        insert_entropy = insert_blk_operator.entropy()

        insert_blk_prob = insert_blk_operator.log_prob(insert_blk_actions)

        one_selected_blk_emb = node_emb[torch.arange(node_emb.shape[0]), insert_blk_actions]
        selected_blk_emb = one_selected_blk_emb.unsqueeze(1).expand(-1, self.num_blk, -1)

        one_org_insert_blk_info = org_blk_info[torch.arange(node_emb.shape[0]), insert_blk_actions]
        org_insert_blk_info = one_org_insert_blk_info.unsqueeze(1).expand(-1, self.num_blk, -1)

        target_logits = self.get_target_blk_logits(graph_info, node_emb[:, :self.num_blk, :], org_blk_info[:, : self.num_blk, :], selected_blk_emb, org_insert_blk_info)

        target_logits = target_logits.unsqueeze(0)

        target_operator = CategoricalMasked(logits=target_logits, masks=target_mask, device=self.device)

        if actions is not None:
            target_blk_actions = actions[:,1]
        else:
            target_blk_actions = target_operator.sample()
        target_blk_prob = target_operator.log_prob(target_blk_actions)
        target_entropy = target_operator.entropy()

        one_target_blk_emb = node_emb[torch.arange(node_emb.shape[0]), target_blk_actions]

        one_target_blk_info = org_blk_info[torch.arange(node_emb.shape[0]), target_blk_actions]

        left_right_mask = left_right_mask[torch.arange(left_right_mask.shape[0]), target_blk_actions]

        left_right_rotate_logits = self.get_left_right_rotate_logits(org_graph_info, one_selected_blk_emb,
                                                                     one_org_insert_blk_info, one_target_blk_emb,
                                                                     one_target_blk_info)
        left_right_rotate_logits = left_right_rotate_logits.unsqueeze(0)
        left_right_rotate_operator = CategoricalMasked(logits=left_right_rotate_logits, masks=left_right_mask,
                                                       device=self.device)

        if actions is not None:
            left_right_rotate_action = actions[:, 2]
        else:
            left_right_rotate_action = left_right_rotate_operator.sample()

        left_right_rotate_logp = left_right_rotate_operator.log_prob(left_right_rotate_action)
        left_right_rotate_entropy = left_right_rotate_operator.entropy()

        logp_sum = left_right_rotate_logp + target_blk_prob + insert_blk_prob
        entropy_sum = left_right_rotate_entropy + target_entropy + insert_entropy

        final_actions = torch.cat([insert_blk_actions.unsqueeze(-1), target_blk_actions.unsqueeze(-1),
                                   left_right_rotate_action.unsqueeze(-1)], -1)

        mask = insert_mask.type(torch.BoolTensor).to(self.device)
        insert_blk_logits = torch.where(mask, insert_blk_logits, torch.tensor(-1e8).to(self.device))

        mask = target_mask.type(torch.BoolTensor).to(self.device)
        target_logits = torch.where(mask, target_logits, torch.tensor(-1e8).to(self.device))

        mask = left_right_mask.type(torch.BoolTensor).to(self.device)
        left_right_rotate_logits = torch.where(mask, left_right_rotate_logits, torch.tensor(-1e8).to(self.device))

        return final_actions, logp_sum, entropy_sum, self.critic( x,  edge_index, edge_attr), [insert_blk_logits, target_logits, left_right_rotate_logits]

    def get_prob(self, x, edge_index, edge_attr, insert_mask, target_mask, left_right_mask, actions=None):

        batch_size = int(x.shape[0] / (self.num_blk + self.num_tml))

        insert_mask = insert_mask.reshape([batch_size, self.num_blk])
        target_mask = target_mask.reshape([batch_size, self.num_blk])
        left_right_mask = left_right_mask.reshape([batch_size, self.num_blk, -1])

        org_graph_info, node_emb = self.get_node_embeddings(x, edge_index, edge_attr)

        org_blk_info = x.reshape([batch_size, -1, 10])

        graph_info = org_graph_info.unsqueeze(1).expand(-1, self.num_blk, -1)

        insert_blk_logits = self.get_insert_blk_logits(graph_info, node_emb[:, :self.num_blk, :],
                                                       org_blk_info[:, : self.num_blk, :])
        insert_blk_operator = CategoricalMasked(logits=insert_blk_logits, masks=insert_mask, device=self.device)
        if actions is not None:
            insert_blk_actions = actions[:, 0]
        else:
            insert_blk_actions = insert_blk_operator.sample()

        one_selected_blk_emb = node_emb[torch.arange(node_emb.shape[0]), insert_blk_actions]
        selected_blk_emb = one_selected_blk_emb.unsqueeze(1).expand(-1, self.num_blk, -1)

        one_org_insert_blk_info = org_blk_info[torch.arange(node_emb.shape[0]), insert_blk_actions]
        org_insert_blk_info = one_org_insert_blk_info.unsqueeze(1).expand(-1, self.num_blk, -1)

        target_logits = self.get_target_blk_logits(graph_info, node_emb[:, :self.num_blk, :],
                                                   org_blk_info[:, : self.num_blk, :], selected_blk_emb,
                                                   org_insert_blk_info)

        target_logits = target_logits.unsqueeze(0)

        target_operator = CategoricalMasked(logits=target_logits, masks=target_mask, device=self.device)

        if actions is not None:
            target_blk_actions = actions[:, 1]
        else:
            target_blk_actions = target_operator.sample()

        one_target_blk_emb = node_emb[torch.arange(node_emb.shape[0]), target_blk_actions]

        one_target_blk_info = org_blk_info[torch.arange(node_emb.shape[0]), target_blk_actions]

        left_right_mask = left_right_mask[torch.arange(left_right_mask.shape[0]), target_blk_actions]

        left_right_rotate_logits = self.get_left_right_rotate_logits(org_graph_info, one_selected_blk_emb,
                                                                     one_org_insert_blk_info, one_target_blk_emb,
                                                                     one_target_blk_info)
        left_right_rotate_logits = left_right_rotate_logits.unsqueeze(0)
        left_right_rotate_operator = CategoricalMasked(logits=left_right_rotate_logits, masks=left_right_mask,
                                                       device=self.device)

        mask = insert_mask.type(torch.BoolTensor).to(self.device)
        insert_blk_logits = torch.where(mask, insert_blk_logits, torch.tensor(-1e8).to(self.device))

        mask = target_mask.type(torch.BoolTensor).to(self.device)
        target_logits = torch.where(mask, target_logits, torch.tensor(-1e8).to(self.device))

        mask = left_right_mask.type(torch.BoolTensor).to(self.device)
        left_right_rotate_logits = torch.where(mask, left_right_rotate_logits, torch.tensor(-1e8).to(self.device))

        return insert_blk_logits, target_logits, left_right_rotate_logits