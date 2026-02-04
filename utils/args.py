from argparse import ArgumentParser
import os

"""
CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
Paper: https://openreview.net/forum?id=86IvZmY26S
Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
         Zhentao Tang, Mingxuan Yuan, Junchi Yan
License: Non-Commercial License (see LICENSE). Commercial use requires permission.
Signature: CORE Authors (NeurIPS 2025)
"""

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--circuit", type=str, default='ami33', help='circuit name')
    parser.add_argument("--result_dir", type=str, default="result-RL_SA-GSRC")
    # parser.add_argument("--result_dir", type=str, default="result-debug-2")
    parser.add_argument("--episode_len", type=int, default=int(50000))
    parser.add_argument("--buffer_size", type=int, default=int(4000))
    parser.add_argument("--total_step", type=int, default=int(200000))
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--num_env", type=int, default=4)
    parser.add_argument("--num_act", type=int, default=3)
    parser.add_argument("--weight_hpwl", type=float, default=1.0)
    parser.add_argument("--weight_area", type=float, default=0.0)
    parser.add_argument("--weight_feedthrough", type=float, default=0.0)

    parser.add_argument("--num_floorplanning_layer", type=int, default=1)
    parser.add_argument("--num_GNN_layer", type=int, default=0)
    parser.add_argument("--num_MLP_layer", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0, help="weight_decay for Adam")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="coefficient for value loss")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="coefficient for entropy loss")
    parser.add_argument("--print_info", type=int, default=0)
    parser.add_argument("--method_construct_graph", type=str, default="both")
    parser.add_argument("--enable_GNN", type=int, default=1)
    parser.add_argument("--enable_CNN", type=int, default=1)
    parser.add_argument("--gnn_type", type=str, default="MyGNN2", choices=["MyGNN1", "MyGNN2"])
    parser.add_argument("--cnn_type", type=str, default="MyCNN", choices=["ResNet18", "MyCNN", "MyCNN2"])

    # parser.add_argument("--method_reward", type=str, default="sparse")
    # parser.add_argument("--method_reward", type=str, default="rnd")
    parser.add_argument("--method_reward", type=str, default="minus_cost_v3")


    parser.add_argument("--enable_restart_policy", type=int, default=1)
    parser.add_argument("--gap_restart_policy", type=int, default=5000)

    

    parser.add_argument("--enable_SA", type=int, default=1)
    parser.add_argument("--init_temperature", type=float, default=1e6)
    parser.add_argument("--temperature_decay", type=float, default=0.6)
    parser.add_argument("--gap_update_temperature", type=int, default=int(1e3), help="how many steps to update temperature")

    parser.add_argument("--enable_tqdm", type=int, default=1)
    parser.add_argument("--enable_draw", type=int, default=1)

    
    

    args = parser.parse_args()

    args.result_dir = os.path.join( args.result_dir, args.circuit )
    args.input_blk = f'input_pa2/{args.circuit}.block'
    args.input_net = f'input_pa2/{args.circuit}.nets'
    
    # args.buffer_size = args.buffer_size // (args.num_env * args.episode_len) * (args.num_env * args.episode_len)

    return args