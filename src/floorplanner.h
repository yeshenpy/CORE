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
    map<string, vector<float>> getNodeInfoMap();
    vector<vector<vector<float>>> constructConnectionMatrix(); //获取邻接矩阵
    Floorplanner(int _num_layer, double _weight_hpwl, double _weight_area, double _weight_feedthrough);
    void parse_blk(const string& input_path);
    void parse_net(const string& input_path);
    void partition();
    void get_blk_left_and_right();
    void initialize_edge_count();
    void Insert_to_target_left_right_rotate(int step, string insert_name, string target_name, int left_or_right_rotate);
    void initialize_node_list();
    void reset();
    void reset_wo_init_tree();
    void initializeFrom(const Floorplanner& other);


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
    string write_report_string(double totaltime);
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


};
#endif 