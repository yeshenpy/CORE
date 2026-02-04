/*
 * CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
 * Paper: https://openreview.net/forum?id=86IvZmY26S
 * Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
 *          Zhentao Tang, Mingxuan Yuan, Junchi Yan
 * License: Non-Commercial License (see LICENSE). Commercial use requires permission.
 * Signature: CORE Authors (NeurIPS 2025)
 */

#ifndef FLOORPLANNER_H
#define FLOORPLANNER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <list>
#include <random>
#include <tuple>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <assert.h>
#include <stack>
#include "block.h"
#include "node.h"
#include "net.h"

using namespace std;


class Floorplanner{
public:
    int chip_w, chip_h;
    int num_blk, num_tml, num_net;
    int num_layer;

    double new_chip_w, new_chip_h;
    map<string, int> edge_count;
    int max_edge = 0;
    // use this to create tsv
    int tsv_name;
    int gap_iter_update_temperature;

    // weight in calculate cost
    double weight_hpwl;
    double weight_area;
    double weight_feedthrough;

    vector<Node*> roots;
    vector<int> x_max_each_layer;
    vector<int> y_max_each_layer;

    // concrete objects
    map<string, Block*> name2blk;
    map<string, Terminal*> name2tml;
    map<string, int> name2index;
    map<string, Node*> name2Node;
    vector<Net*>      net_list;
    // abstract
    vector<Node*>    node_list;
    
    random_device gen;
    vector<list<ContourNode>> contour_lines;

    vector<vector<int>> edge_info();
    vector<vector<double>> edge_attr();
    map<string, vector<float>> getBlkInfoMap();
    map<string, vector<float>> getTmlInfoMap();


    vector<vector<vector<float>>> constructConnectionMatrix(); //获取邻接矩阵
    Floorplanner(int _num_layer, double _weight_hpwl, double _weight_area, double _weight_feedthrough);


    void initializeFrom_info( map<string, vector<int>> other_name2blk,   map<string, vector<string>> other_nodeinfomap, vector<string> otherroots,   vector<int> other_x_max_each_layer,   vector<int> other_y_max_each_layer);



    vector<float> Graph_feature();
    void parse_blk(const string& input_path);
    void parse_net(const string& input_path);
    void partition();
    void get_blk_left_and_right();
    void initialize_edge_count();
    void Insert_to_target_left_right_rotate(int step, string insert_name, string target_name, int left_or_right_rotate);
    void initialize_node_list();
    void reset(int seed);
    void reset_wo_init_tree();
    void initializeFrom(const Floorplanner& other);
    string write_report_string(double totaltime);
    void initializeFrom_other_FP(const Floorplanner& other);

    list<tuple<string, string, int, double, double>> get_place_sequence();
    double SA_HPWL();
    double HPWL() ;
    double calculate_area();
    double calculate_cost();
    double updcontour(Node* node);
    int coordinate(Node* node, int order=0);
    void get_all_nodes_coordinate();

    tuple<int, int, int, int, Node*, Node*> RL_delandins(int del, int ins, int left_or_right);
    tuple<int, int, int, int, Node*, Node*> delandins(int del, int ins);
    void revert_delandins(int del_node, int ins_node, int prev_lorr, int child_lorr, Node* prevblk, Node* childblk);
    void swap(int node1, int node2);
    void rotate(int node_idx);
    int insert_node_to_tree(int node_src, int node_dst, int insert_to_right);
    //void write_report();
    void write_report(double totaltime,const string& output_path);
    tuple<Node*, int, Node*, Node*> move_tree(int node_del, int node_ins);
    void revert_move_tree(Node* original_parent, int is_original_left, Node* node_del, Node* node_ins);

    vector<vector<int>> get_blk_feat();
    map<string, int> get_name2idx();
    vector<vector<int>> get_graph_src_dst_from_tree();
    vector<vector<int>> get_graph_src_dst_from_netlist();
    vector<int> get_all_root_ids();
    vector<int> get_nodes_with_two_children_ids();

    int get_x_max() const;
    int get_y_max() const;
    map<string,double> summary(int iteration_index);

    vector<int> get_subtree_node_indices(int node_idx);
    void get_subtree_node_indices__(int node_idx, vector<int>& subtree_node_indices, map<string,int>&name2idx);
    int is_adjacent(Block* b1, Block* b2);
    int calculate_feedthrough();
    void set_roots(vector<int>);

    double calculate_outbound();

    Node* findCorrespondingNode(const Node* otherNode) {
    for (const auto& node : node_list) {
        if (node->_name == otherNode->_name) {
            return node;
        }
    }
    return nullptr;  // 如果找不到对应的 Node，则返回 nullptr
    }

    Floorplanner(const Floorplanner& other) {
        // Copy simple types
        chip_w = other.chip_w;
        chip_h = other.chip_h;
        num_blk = other.num_blk;
        num_tml = other.num_tml;
        num_net = other.num_net;
        num_layer = other.num_layer;
        new_chip_w = other.new_chip_w;
        new_chip_h = other.new_chip_h;
        edge_count = other.edge_count;
        max_edge = other.max_edge;
        tsv_name = other.tsv_name;
        gap_iter_update_temperature = other.gap_iter_update_temperature;
        weight_hpwl = other.weight_hpwl;
        weight_area = other.weight_area;
        weight_feedthrough = other.weight_feedthrough;


        contour_lines = vector<list<ContourNode>>(num_layer);
        // Deep copy vectors

        x_max_each_layer = other.x_max_each_layer;
        y_max_each_layer = other.y_max_each_layer;

        // Deep copy maps
        name2blk.clear();
        for (const auto& entry : other.name2blk) {
            name2blk[entry.first] = new Block(*(entry.second));
        }

        name2tml.clear();
        for (const auto& entry : other.name2tml) {
            name2tml[entry.first] = new Terminal(*(entry.second));
        }

        name2index = other.name2index;


        // Deep copy vector of pointers
        net_list.clear();
        for (const auto& net : other.net_list) {
            net_list.push_back(new Net(*net));
        }

        node_list.clear();
        name2Node.clear();
        roots.clear();

        map<const Node*, Node*> nodeMap;  // 用于映射已经拷贝的节点

        for (const auto& node : other.node_list) {
            Node* newNode = new Node(*node);
            nodeMap[node] = newNode;
        }

        // 然后处理连接关系
        for (const auto& node : other.node_list) {
            Node* newNode = nodeMap[node];

            // 处理连接关系
            if (node->_prev) {
                newNode->_prev = nodeMap[node->_prev];
            }
            if (node->_left) {
                newNode->_left = nodeMap[node->_left];
            }
            if (node->_right) {
                newNode->_right = nodeMap[node->_right];
            }

            node_list.push_back(newNode);
        }

        // 深拷贝 name2Node
        for (const auto& entry : other.name2Node) {
            name2Node[entry.first] = nodeMap[entry.second];
        }

        // 深拷贝 roots，并维护连接关系
        for (const auto& root : other.roots) {
            roots.push_back(nodeMap[root]);
        }

    }
    Floorplanner DeepCopy() const {
        return Floorplanner(*this);
    }


};
#endif 