/*
 * CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
 * Paper: https://openreview.net/forum?id=86IvZmY26S
 * Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
 *          Zhentao Tang, Mingxuan Yuan, Junchi Yan
 * License: Non-Commercial License (see LICENSE). Commercial use requires permission.
 * Signature: CORE Authors (NeurIPS 2025)
 */

#include "block.h"
#include <cstdlib>

Block::Block(const string& name, int w, int h, BlockType _block_type, int _layer, int _on_FP, int _left, int _right) :
_name(name), _w(w), _h(h), _x(0), _y(0), order(0), block_type(_block_type), layer(_layer),  ratio_min(0.0), ratio_max(0.0), on_FP(_on_FP), left(_left), right(_right){
}

void Block::rotate(){
    int tmp = _w;
    _w = _h;
    _h = tmp;
}


string Block::display() const {
    return "<Block, name = " + _name + ", w = " + to_string(_w) + ", h = " + to_string(_h) + ", x = " + to_string(_x) + ", y = " + to_string(_y) + ", layer = " + to_string(layer) + ">";
}





Terminal::Terminal(const string& name, double x, double y, int _layer):_name(name), _x(x), _y(y), layer(_layer) {

}

Terminal::~Terminal(){

}

string Terminal::display() const{
    return "<Terminal, name = " + _name + ", x = " + to_string(_x) + ", y = " + to_string(_y) + ">";
}