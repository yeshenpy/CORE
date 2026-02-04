/*
 * CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
 * Paper: https://openreview.net/forum?id=86IvZmY26S
 * Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
 *          Zhentao Tang, Mingxuan Yuan, Junchi Yan
 * License: Non-Commercial License (see LICENSE). Commercial use requires permission.
 * Signature: CORE Authors (NeurIPS 2025)
 */

#include "node.h"

Node::Node(const string& name):_name(name), _prev(nullptr), _right(nullptr), _left(nullptr) {

}


string Node::display()const{
    string p,l,r;
    p = (_prev==nullptr)? "None": _prev->_name;
    l = (_left==nullptr)? "None": _left->_name;
    r = (_right==nullptr)? "None": _right->_name;
    return "<Node, name = " + _name + ", prev = " + p + ", left = " + l + ", right = " + r + ">";
}


ContourNode::ContourNode(double x1, double x2, double y): _x1(x1), _x2(x2), _y(y){

}
