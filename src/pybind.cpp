#include "floorplanner.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;


PYBIND11_MODULE(tree, m){
    py::class_<Node>(m, "Node")
        .def(py::init<string>())
        .def("__repr__", &Node::display)
        .def_readwrite("name", &Node::_name)
        .def_readwrite("prev", &Node::_prev)
        .def_readwrite("right", &Node::_right)
        .def_readwrite("left", &Node::_left)
    ;


    py::class_<Block>(m, "Block")
        .def(py::init<string&, int, int>())
        .def("rotate", &Block::rotate)
        .def("__repr__", &Block::display)
        .def_readwrite("name", &Block::_name)
        .def_readwrite("w", &Block::_w)
        .def_readwrite("h", &Block::_h)
        .def_readwrite("x", &Block::_x)
        .def_readwrite("y", &Block::_y)
        .def_readwrite("layer", &Block::layer)
        .def_readwrite("left", &Block::left)
        .def_readwrite("right", &Block::right)
        .def_readwrite("on_FP", &Block::on_FP)
        .def_readwrite("order", &Block::order)
    ;



    py::class_<Terminal>(m, "Terminal")
        .def(py::init<string&, int, int>())
        .def("__repr__", &Terminal::display)
        .def_readwrite("name", &Terminal::_name)
        .def_readwrite("x", &Terminal::_x)
        .def_readwrite("y", &Terminal::_y)
        .def_readwrite("layer", &Terminal::layer)
    ;

    py::class_<Net>(m, "Net")
        .def(py::init<>())
        .def("__repr__", &Net::display)
        .def_readwrite("cell_list", &Net::cell_list)
    ;

    
    py::class_<Floorplanner>(m, "Floorplanner")
        .def(py::init<int, double, double, double>())
        .def("parse_blk", &Floorplanner::parse_blk)
        .def("parse_net", &Floorplanner::parse_net)
        .def("initialize_node_list", &Floorplanner::initialize_node_list)
        .def("initializeFrom", &Floorplanner::initializeFrom)
        .def("initializeFrom_other_FP", &Floorplanner::initializeFrom_other_FP)
        .def("getBlkInfoMap", &Floorplanner::getBlkInfoMap)
        .def("getTmlInfoMap", &Floorplanner::getTmlInfoMap)

        .def("constructConnectionMatrix", &Floorplanner::constructConnectionMatrix)
        .def("write_report_string", &Floorplanner::write_report_string)
        .def("Graph_feature", &Floorplanner::Graph_feature)
        .def("get_place_sequence", &Floorplanner::get_place_sequence)
        .def("SA_HPWL", &Floorplanner::SA_HPWL)
        .def("HPWL", &Floorplanner::HPWL)
        .def("calculate_area", &Floorplanner::calculate_area)
        .def("calculate_cost", &Floorplanner::calculate_cost)
        .def("calculate_outbound", &Floorplanner::calculate_outbound)

        .def("initializeFrom_info", &Floorplanner::initializeFrom_info)


        .def("Insert_to_target_left_right_rotate",  &Floorplanner::Insert_to_target_left_right_rotate)
        .def("rotate", &Floorplanner::rotate)
        .def("swap", &Floorplanner::swap)
        .def("move_tree", &Floorplanner::move_tree, py::return_value_policy::reference)
        .def("revert_move_tree", &Floorplanner::revert_move_tree)
        .def("delandins", &Floorplanner::delandins, py::return_value_policy::reference)
        .def("RL_delandins", &Floorplanner::RL_delandins, py::return_value_policy::reference)

        .def("revert_delandins", &Floorplanner::revert_delandins)
        .def("insert_node_to_tree", &Floorplanner::insert_node_to_tree)


        .def("edge_info", &Floorplanner::edge_info)
        .def("edge_attr", &Floorplanner::edge_attr)
        .def("DeepCopy", &Floorplanner::DeepCopy)

        .def("get_all_nodes_coordinate", &Floorplanner::get_all_nodes_coordinate)
        .def("write_report", &Floorplanner::write_report)
        .def("summary", &Floorplanner::summary)
        .def("get_blk_feat", &Floorplanner::get_blk_feat)
        .def("get_graph_src_dst_from_tree", &Floorplanner::get_graph_src_dst_from_tree)
        .def("get_graph_src_dst_from_netlist", &Floorplanner::get_graph_src_dst_from_netlist)
        .def("get_all_root_ids", &Floorplanner::get_all_root_ids)
        .def("get_nodes_with_two_children_ids", &Floorplanner::get_nodes_with_two_children_ids)
        .def("reset", &Floorplanner::reset)
        .def("reset_wo_init_tree", &Floorplanner::reset_wo_init_tree)
        
        .def("get_subtree_node_indices", &Floorplanner::get_subtree_node_indices)
        .def("is_adjacent", &Floorplanner::is_adjacent)
        .def("calculate_feedthrough", &Floorplanner::calculate_feedthrough)
        .def("get_x_max", &Floorplanner::get_x_max)
        .def("get_y_max", &Floorplanner::get_y_max)
        .def("set_roots", &Floorplanner::set_roots)

        

        .def_readwrite("new_chip_w", &Floorplanner::new_chip_w)
        .def_readwrite("new_chip_h", &Floorplanner::new_chip_h)
        .def_readwrite("chip_w", &Floorplanner::chip_w)
        .def_readwrite("chip_h", &Floorplanner::chip_h)
        .def_readwrite("num_blk", &Floorplanner::num_blk)
        .def_readwrite("num_tml", &Floorplanner::num_tml)
        .def_readwrite("num_net", &Floorplanner::num_net)
        .def_readwrite("x_max_each_layer", &Floorplanner::x_max_each_layer)
        .def_readwrite("y_max_each_layer", &Floorplanner::y_max_each_layer)
        .def_readwrite("node_list", &Floorplanner::node_list)
        .def_readwrite("net_list", &Floorplanner::net_list)
        .def_readwrite("name2blk", &Floorplanner::name2blk)
        .def_readwrite("name2tml", &Floorplanner::name2tml)
        .def_readwrite("num_layer", &Floorplanner::num_layer)
        .def_readwrite("roots", &Floorplanner::roots)
        .def_readwrite("name2index", &Floorplanner::name2index)
    ;

}

