/*
 * CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
 * Paper: https://openreview.net/forum?id=86IvZmY26S
 * Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
 *          Zhentao Tang, Mingxuan Yuan, Junchi Yan
 * License: Non-Commercial License (see LICENSE). Commercial use requires permission.
 * Signature: CORE Authors (NeurIPS 2025)
 */

#include "net.h"

string Net::display()const{
    string res = "";
    res += "Net([";
    if(cell_list.size() > 0){
        res += cell_list[0];
    }
    for(uint i=1; i<cell_list.size(); ++i){
        res += ", ";
        res += cell_list[i];
    }
    res += "])";
    return res;
}

