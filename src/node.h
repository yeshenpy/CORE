#ifndef NODE_H
#define NODE_H
#include <string>
using namespace std;

class Node{
public:
    string    _name;
	Node*    _prev;
	Node*    _right;
	Node*    _left;

	Node(const string& name);
    string display()const;

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