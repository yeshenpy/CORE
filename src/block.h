/*
 * CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
 * Paper: https://openreview.net/forum?id=86IvZmY26S
 * Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
 *          Zhentao Tang, Mingxuan Yuan, Junchi Yan
 * License: Non-Commercial License (see LICENSE). Commercial use requires permission.
 * Signature: CORE Authors (NeurIPS 2025)
 */

#ifndef BLOCK_H
#define BLOCK_H
#include <iostream>
#include <string>
#include "node.h"
using namespace std;

enum BlockType {HARD, SOFT, TSV};

class Block{
public:

    string          _name;
    int          _w;
    int          _h;
    int          _x;
    int          _y;
    int          order;


    BlockType       block_type;
    int          layer;
    double          ratio_min;
    double          ratio_max;
    int          on_FP;
    int          left;
    int          right;


    Block(const string& name, int w, int h, BlockType _block_type=HARD, int _layer=0, int _on_FP=0, int _left=0, int _right=0);
    void rotate();
    string display() const;
};


class Terminal{
public:
    string      _name;
    double      _x;
    double      _y;
    int         layer;

    Terminal(const string& name, double x, double y, int _layer=0);
    ~Terminal();
    string display()const;
};



#endif