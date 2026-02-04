/*
 * CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
 * Paper: https://openreview.net/forum?id=86IvZmY26S
 * Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
 *          Zhentao Tang, Mingxuan Yuan, Junchi Yan
 * License: Non-Commercial License (see LICENSE). Commercial use requires permission.
 * Signature: CORE Authors (NeurIPS 2025)
 */

#ifndef NODE_H
#define NODE_H
#include <string>
using namespace std;

//
//class NodeInfo {
//public:
//    string _insert;
//    string _target;
//    int _left_right;
//    double _width;
//    double _height;
//
//    NodeInfo(const std::string& insert, const std::string& target, const std::string& left_right, double w, double h)
//        : _insert(insert), _target(target), parentNode(parent), width(w), height(h) {}
//};



class Node{
public:
    string    _name;
	Node*    _prev;
	Node*    _right;
	Node*    _left;

	Node(const string& name);
    string display()const;

    Node(const Node& other) {
    _name = other._name;
    _prev = nullptr;
    _right = nullptr;
    _left = nullptr;
    }

    Node& operator=(const Node& other) {
    if (this != &other) {
        _name = other._name;
        _prev = nullptr;
        _right = nullptr;
        _left = nullptr;
    }
    return *this;
    }

    Node* clone() const {
        return new Node(*this);
    }


};


class ContourNode {
public:
	double _x1, _x2, _y;
    ContourNode(double x1, double x2, double y);

    ContourNode clone() const {
        return ContourNode(*this);  // 这里使用拷贝构造函数来实现深拷贝
    }

};


#endif