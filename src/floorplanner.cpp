/*
 * CORE: Collaborative Optimization with Reinforcement Learning and Evolutionary Algorithm for Floorplanning
 * Paper: https://openreview.net/forum?id=86IvZmY26S
 * Authors: Pengyi Li, Shixiong Kai, Jianye Hao, Ruizhe Zhong, Hongyao Tang,
 *          Zhentao Tang, Mingxuan Yuan, Junchi Yan
 * License: Non-Commercial License (see LICENSE). Commercial use requires permission.
 * Signature: CORE Authors (NeurIPS 2025)
 */

#include "floorplanner.h"
#include <random>
#include <cmath>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <stdexcept>
using namespace std;


void  Floorplanner::get_blk_left_and_right(){

    for(size_t i=0; i<node_list.size(); ++i){
        auto name = node_list[i]->_name;
        if(node_list[i]->_left == nullptr){
            name2blk[name]->left = 0;
        }else{
            name2blk[name]->left = 1;
        }


        if (node_list[i]->_right == nullptr){
            name2blk[name]->right = 0;
        }
        else{
            name2blk[name]->right = 1;
        }


//        node_list[i]->_left = nullptr;
//        node_list[i]->_right = nullptr;
//        node_list[i]->_prev = nullptr;
    }

}



vector<float> Floorplanner::Graph_feature() {

    vector<float> graph_feature;

    for (const auto& entry : name2blk) {
        graph_feature.push_back(double(entry.second->_x)/new_chip_w);
        graph_feature.push_back(double(entry.second->_y)/new_chip_h);
    }

    return graph_feature;
}





std::map<std::string, std::vector<float>> Floorplanner::getBlkInfoMap() {
    //cout<< 5 << endl;
    get_blk_left_and_right();
    //cout<< 6 << endl;
    std::map<std::string, std::vector<float>> nodeInfoMap;
    // 处理 name2blk
    for (const auto& entry : name2blk) {
        std::string name = entry.first;
        Block* b_temp = entry.second;
        std::vector<float> nodeInfo(10, 0);

        //cout<< 10 << b_temp->on_FP << endl;
        if(b_temp->on_FP==0)
        {
//            cout<< 11  << endl;
            nodeInfo[0] = 0.0;
            nodeInfo[1] = 0.0;
            nodeInfo[2] =  double(b_temp->_w)/new_chip_w;  // 宽度
            nodeInfo[3] =  double(b_temp->_h)/new_chip_h;  // 高度
            nodeInfo[4] =  0.0;
            nodeInfo[5] =  0.0;
            nodeInfo[6] =  0.0;
            nodeInfo[7] =  0.0;
            nodeInfo[8] =  0.0;
            nodeInfo[9] =  0.0;
        }else{
            nodeInfo[0] =  double(b_temp->_x)/new_chip_w;  // x 坐标
            nodeInfo[1] =  double(b_temp->_y)/new_chip_h;  // y 坐标
            nodeInfo[2] =  double(b_temp->_w)/new_chip_w;  // 宽度
            nodeInfo[3] =  double(b_temp->_h)/new_chip_h;  // 高度
            nodeInfo[4] =  double(b_temp->_x + b_temp->_w)/new_chip_w;
            nodeInfo[5] =  double(b_temp->_y + b_temp->_h)/new_chip_h;
            nodeInfo[6] =  1.0;
            nodeInfo[7] =  double(b_temp->order+1.0)/num_blk;
            nodeInfo[8] =  double(b_temp->left);
            nodeInfo[9] =  double(b_temp->right);
//            cout<< 12  << endl;
        }
//        cout << 13<<endl;

//        nodeInfo[6] = 1;         // 标记类型，1为节点，0为 tml
//        nodeInfo[7] = 0;         // 标记类型，1为节点，0为 tml

        // 设置是否在画布上的标记
//        if (name2node.count(name) != 0) {
//            Node* node = name2node[name];
//            nodeInfo[0] = node->_x;  // x 坐标
//            nodeInfo[1] = node->_y;  // y 坐标
//            nodeInfo[4] = 1;         // 标记类型，1为节点，0为 tml
//            nodeInfo[5] = node->_x + node->_w;
//            nodeInfo[6] = node->_y + node->_h;
//        }
        nodeInfoMap[name] = nodeInfo;
    }
//    cout<< 20  << endl;
//    // 处理 name2tml
//    for (const auto& entry : name2tml) {
//        std::string name = entry.first;
//        Terminal* t_temp = entry.second;
//        std::vector<float> nodeInfo(10, 0);
//        nodeInfo[0] =  static_cast<double>(t_temp->_x)/new_chip_w;  // 宽度
//        nodeInfo[1] =  static_cast<double>(t_temp->_y)/new_chip_h;  // 高度
//        nodeInfo[6] =  1.0;
//        // 设置是否为 Terminal 的标记
////        nodeInfo[6] = 0;         // 标记类型，1为节点，0为 tml
////        nodeInfo[7] = 1;         // 标记类型，1为节点，0为 tml
//        nodeInfoMap[name] = nodeInfo;
//    }

    return nodeInfoMap;
}



std::map<std::string, std::vector<float>> Floorplanner::getTmlInfoMap() {
    //cout<< 5 << endl;
    //get_blk_left_and_right();
    //cout<< 6 << endl;
    std::map<std::string, std::vector<float>> nodeInfoMap;
    // 处理 name2blk
    for (const auto& entry : name2tml) {
        std::string name = entry.first;
        Terminal* t_temp = entry.second;
        std::vector<float> nodeInfo(10, 0);
        nodeInfo[0] =  static_cast<double>(t_temp->_x)/new_chip_w;  // 宽度
        nodeInfo[1] =  static_cast<double>(t_temp->_y)/new_chip_h;  // 高度
        nodeInfo[6] =  1.0;
        // 设置是否为 Terminal 的标记
//        nodeInfo[6] = 0;         // 标记类型，1为节点，0为 tml
//        nodeInfo[7] = 1;         // 标记类型，1为节点，0为 tml
        nodeInfoMap[name] = nodeInfo;
    }

    return nodeInfoMap;
}



vector<vector<int>> Floorplanner::edge_info() {
    // 初始化连接矩阵

    vector<vector<int>> edge_list;
    // 遍历每个网络
    for (const auto& net : net_list) {
        // 对网络中的节点两两配对，标记连接关系
        for (size_t i = 0; i < net->cell_list.size(); ++i) {
            for (size_t j = i + 1; j < net->cell_list.size(); ++j) {
                // 获取节点在 name2index 中的索引
                int nodeIndex1 = name2index[net->cell_list[i]];
                int nodeIndex2 = name2index[net->cell_list[j]];
                edge_list.push_back({nodeIndex1, nodeIndex2});
                edge_list.push_back({nodeIndex2, nodeIndex1});

            }
        }
    }
    // 返回连接矩阵
    return edge_list;
}



vector<vector<double>> Floorplanner::edge_attr() {
    // 初始化连接矩阵

    vector<vector<double>> edge_attr_list;
    // 遍历每个网络
    for (const auto& net : net_list) {
        // 对网络中的节点两两配对，标记连接关系
        for (size_t i = 0; i < net->cell_list.size(); ++i) {
            for (size_t j = i + 1; j < net->cell_list.size(); ++j) {
                // 获取节点在 name2index 中的索引
//                int nodeIndex1 = name2index[net->cell_list[i]];
//                int nodeIndex2 = name2index[net->cell_list[j]];
//                double X_1;
//                double Y_1;
//                double X_2;
//                double Y_2;
                int on_the_FP = 1;


               // cout<< "1111"<<endl;

                if (name2blk.count(net->cell_list[i]))
                {
                    if(name2blk[net->cell_list[i]]->on_FP == 0){

                        on_the_FP = 0;
                    }

                }

                if (name2blk.count(net->cell_list[j]))
                {
                    if( name2blk[net->cell_list[j]]->on_FP == 0)
                    {
                        on_the_FP = 0;
                    }

                }
                int nodeIndex1 = name2index[net->cell_list[i]];
                int nodeIndex2 = name2index[net->cell_list[j]];
                string new_name_one = nodeIndex1 + "_" + nodeIndex2;

                if (on_the_FP==1)
                {

//                    if (name2blk.count( net->cell_list[i])) {
//
//                        Block* curblock = name2blk[net->cell_list[i]];
//                        X_1 = double(curblock->_x) + double(curblock->_w)/double(2);
//                        Y_1 = double(curblock->_y) + double(curblock->_h)/double(2);
//
//                    }
//                    // this is a terminal
//                    else{
//                        Terminal* curTerminal = name2tml[net->cell_list[i]];
//                        X_1 = curTerminal->_x;
//                        Y_1 = curTerminal->_y;
//                    }
//                    if (name2blk.count( net->cell_list[j])) {
//                        Block* curblock = name2blk[net->cell_list[j]];
//                        X_2 = double(curblock->_x) + double(curblock->_w)/double(2);
//                        Y_2 = double(curblock->_y) + double(curblock->_h)/double(2);
//                    }
//                    // this is a terminal
//                    else {
//                        Terminal* curTerminal = name2tml[net->cell_list[j]];
//                        X_2 = curTerminal->_x;
//                        Y_2 = curTerminal->_y;
//                    }
//
//                    double X_difference = (X_1 - X_2)/new_chip_w;
//                    double Y_difference = (Y_1 - Y_2)/new_chip_h;

//                    cout << X_difference << Y_difference << edge_count[new_name_one]/double(max_edge) << endl;

                    edge_attr_list.push_back({1.0, edge_count[new_name_one]/double(max_edge)});
                    edge_attr_list.push_back({1.0, edge_count[new_name_one]/double(max_edge)});

                 //   cout<< "33333"<<endl;
                }else{
                //    cout<< "44"<<endl;
                    edge_attr_list.push_back({0.0, edge_count[new_name_one]/double(max_edge)});
                    edge_attr_list.push_back({0.0, edge_count[new_name_one]/double(max_edge)});
                //    cout<< "4444"<<endl;
                }


            }
        }
    }
    // 返回连接矩阵
    return edge_attr_list;
}



vector<vector<vector<float>>> Floorplanner::constructConnectionMatrix() {
    // 初始化连接矩阵
    vector<vector<float>> connectionMatrix(name2index.size(), vector<float>(name2index.size(), 0));
    // 权重矩阵
    vector<vector<float>> XweightMatrix(name2index.size(), vector<float>(name2index.size(), 0));
    vector<vector<float>> YweightMatrix(name2index.size(), vector<float>(name2index.size(), 0));

    // 遍历每个网络
    for (const auto& net : net_list) {
        // 对网络中的节点两两配对，标记连接关系
        for (size_t i = 0; i < net->cell_list.size(); ++i) {
            for (size_t j = i + 1; j < net->cell_list.size(); ++j) {
                // 获取节点在 name2index 中的索引


                int nodeIndex1 = name2index[net->cell_list[i]];
                int nodeIndex2 = name2index[net->cell_list[j]];


                double X_1;
                double Y_1;
                double X_2;
                double Y_2;

                if (name2blk.count( net->cell_list[i])) {
                    Block* curblock = name2blk[net->cell_list[i]];
                    X_1 = double(curblock->_x) + double(curblock->_w)/double(2);
                    Y_1 = double(curblock->_y) + double(curblock->_h)/double(2);
                }
                // this is a terminal
                else{
                    Terminal* curTerminal = name2tml[net->cell_list[i]];
                    X_1 = curTerminal->_x;
                    Y_1 = curTerminal->_y;
                }

                if (name2blk.count( net->cell_list[j])) {
                    Block* curblock = name2blk[net->cell_list[j]];
                    X_2 = double(curblock->_x) + double(curblock->_w)/double(2);
                    Y_2 = double(curblock->_y) + double(curblock->_h)/double(2);
                }
                // this is a terminal
                else {
                    Terminal* curTerminal = name2tml[net->cell_list[j]];
                    X_2 = curTerminal->_x;
                    Y_2 = curTerminal->_y;
                }


                double X_difference = (X_1 - X_2)/new_chip_w;
                double Y_difference = (Y_1 - Y_2)/new_chip_h;

                // 标记连接关系
                connectionMatrix[nodeIndex1][nodeIndex2] = 1;
                connectionMatrix[nodeIndex2][nodeIndex1] = 1;
                XweightMatrix[nodeIndex1][nodeIndex2] = X_difference;
                XweightMatrix[nodeIndex2][nodeIndex1] = -X_difference;
                YweightMatrix[nodeIndex1][nodeIndex2] = Y_difference;
                YweightMatrix[nodeIndex2][nodeIndex1] = -Y_difference;
            }
        }
    }
    for (std::map<std::__cxx11::basic_string<char>, int>::size_type i = 0; i < name2index.size(); ++i) {
        connectionMatrix[i][i] = 1e-10;
    }
    // 返回连接矩阵
    return {connectionMatrix, XweightMatrix, YweightMatrix};
}



void Floorplanner::initializeFrom_info(map<string, vector<int>> other_name2blk,   map<string, vector<string>> other_nodeinfomap, vector<string> otherroots,   vector<int> otherx_max_each_layer,   vector<int> othery_max_each_layer){


        for (const auto& pair : other_name2blk) {
            const string& name = pair.first;
            const vector<int>& vec = pair.second;
            name2blk[name]->_x = vec[0];
            name2blk[name]->_y = vec[1];
            name2blk[name]->_w = vec[2];
            name2blk[name]->_h = vec[3];
            name2blk[name]->order = vec[4];
            name2blk[name]->layer = vec[5];
         }



//        for(auto o :  other_name2blk){
//
//            string name =  o.first;
//  //          cout<< "!!!!!  target name " <<  name<<    " our name "  <<name2blk[name]->_name  << endl;
//            name2blk[name]->_x = o.second->_x;
//            name2blk[name]->_y = o.second->_y;
//            name2blk[name]->_w = o.second->_w;
//            name2blk[name]->_h = o.second->_h;
//            name2blk[name]->order = o.second->order;
//            name2blk[name]->block_type = o.second->block_type;
//            name2blk[name]->layer = o.second->layer;
//            name2blk[name]->ratio_min = o.second->ratio_min;
//            name2blk[name]->ratio_max = o.second->ratio_max;
//        }



//        map<string, Node*> otherNodeMap;
//        for (const auto& otherNode : other_node_list) {
//            otherNodeMap[otherNode->_name] = otherNode;
//        }

        map<string, Node*> ourNodeMap;
        for (const auto& ourNode : node_list) {
            ourNodeMap[ourNode->_name] = ourNode;
        }

        if (node_list.size() == 0){
            cout << "!!!!!! fail node_list size == 0 " << endl;
            std::exit(EXIT_FAILURE);
        }

        // 更新 current 中的节点列表，确保 _prev、_right、_left 的映射关系相同
        for (auto& currentNode : node_list) {

            string current_name = currentNode->_name;

            const vector<string>&  string_vector = other_nodeinfomap[current_name];

//            cout <<"!!!!!!!!!?????????????" << current_name <<   " __ target " << otherNode->_name <<endl;

            if (string_vector[0] == "")
            {
                currentNode->_prev = nullptr;
            }
            else {
                currentNode->_prev = ourNodeMap[string_vector[0]];
            }
            if (string_vector[1] == "")
            {
                currentNode->_right = nullptr;
            }
            else {
                currentNode->_right = ourNodeMap[string_vector[1]];
            }

            if (string_vector[2] == "")
            {
                currentNode->_left = nullptr;
            }
            else {
                currentNode->_left = ourNodeMap[string_vector[2]];
            }
        }

       // cout << "?????" << roots.size()  <<  node_list.size()<< endl;
        if  (roots.size() == 0)
        {
                roots[0] = ourNodeMap[otherroots[0]];
        }
        else{
            for (size_t i = 0; i < roots.size(); ++i) {

              //   cout << "!!!" << otherroots[i] << endl;
                 roots[i] = ourNodeMap[otherroots[i]];
            }
        }

        x_max_each_layer = otherx_max_each_layer;
        y_max_each_layer = othery_max_each_layer;
}

void Floorplanner::initializeFrom(const Floorplanner& other){


        for(auto o :  other.name2blk){

            string name =  o.first;
  //          cout<< "!!!!!  target name " <<  name<<    " our name "  <<name2blk[name]->_name  << endl;
            name2blk[name]->_x = o.second->_x;
            name2blk[name]->_y = o.second->_y;
            name2blk[name]->_w = o.second->_w;
            name2blk[name]->_h = o.second->_h;
            name2blk[name]->order = o.second->order;
            name2blk[name]->block_type = o.second->block_type;
            name2blk[name]->layer = o.second->layer;
            name2blk[name]->ratio_min = o.second->ratio_min;
            name2blk[name]->ratio_max = o.second->ratio_max;
        }


        map<string, Node*> otherNodeMap;
        for (const auto& otherNode : other.node_list) {
            otherNodeMap[otherNode->_name] = otherNode;
        }
        map<string, Node*> ourNodeMap;
        for (const auto& ourNode : node_list) {
            ourNodeMap[ourNode->_name] = ourNode;
        }

        // 更新 current 中的节点列表，确保 _prev、_right、_left 的映射关系相同
        for (auto& currentNode : node_list) {

            string current_name = currentNode->_name;

            Node* otherNode = otherNodeMap[current_name];
//            cout <<"!!!!!!!!!?????????????" << current_name <<   " __ target " << otherNode->_name <<endl;

            if (otherNode->_prev == nullptr)
            {
                currentNode->_prev = nullptr;
            }
            else {
                currentNode->_prev = ourNodeMap[otherNode->_prev->_name];
            }
            if (otherNode->_right == nullptr)
            {
                currentNode->_right = nullptr;
            }
            else {
                currentNode->_right = ourNodeMap[otherNode->_right->_name];
            }

            if (otherNode->_left == nullptr)
            {
                currentNode->_left = nullptr;
            }
            else {
                currentNode->_left = ourNodeMap[otherNode->_left->_name];
            }

        }

        for (size_t i = 0; i < roots.size(); ++i) {

             roots[i] = ourNodeMap[other.roots[i]->_name];
        }

        x_max_each_layer = other.x_max_each_layer;
        y_max_each_layer = other.y_max_each_layer;
        // 按照other.name2blk 修改  name2blk里面的_x,_y, _w,_h,order,block_type,layer,ratio_min,ratio_max 的信息
        // 按照other.node_list， 修改node_list中之中的链接关系， 例如每个节点_prev，   _right，_left是否相同为空，name是否相同，除此之外，是_prev，   _right，_left的name是否相同
}

Floorplanner::Floorplanner(int _num_layer, double _weight_hpwl, double _weight_area, double _weight_feedthrough){
    num_layer = _num_layer;

    tsv_name = 0;
    // printf("number of layers = %d\n", num_layer);
    roots = vector<Node*>(num_layer, nullptr);
    contour_lines = vector<list<ContourNode>>(num_layer);
    x_max_each_layer = vector<int>(num_layer, -1);
    y_max_each_layer = vector<int>(num_layer, -1);

    // weight in calculate cost
    weight_hpwl = _weight_hpwl;
    weight_area = _weight_area;
    weight_feedthrough = _weight_feedthrough;
}

int Floorplanner::get_x_max() const{
    return (*max_element(x_max_each_layer.begin(), x_max_each_layer.end()));
}

int Floorplanner::get_y_max() const{
    return (*max_element(y_max_each_layer.begin(), y_max_each_layer.end()));
}


void Floorplanner::parse_blk(const string& input_path){
    fstream input;
    input.open(input_path, ios::in);
    string trash, str;
	input >> trash; 
	input >> str;	
    chip_w = stod(str);
	
	input >> str;	
    chip_h = stod(str);
    
	input >> trash; 
	input >> str;	
    num_blk = stod(str);
	
	input >> trash; 
	input >> str;	
    num_tml = stod(str);

	double total_area = 0.0;

	for(int i=0; i<num_blk; ++i){
		string name;
		double w;
		double h;
		input >> str;	
        name = str;
		input >> str;  
        w=stod(str);
		input >> str;  
        h=stod(str);
        name2blk[name] = new Block(name, w, h);
        name2index[name] = i;
        total_area = total_area + double(w)*double(h);
	}
    new_chip_w = pow(double(total_area)*1.1, 0.5);
    new_chip_h = pow(double(total_area)*1.1, 0.5);
    
    
      
  double max_w = 0.0;
  double max_h = 0.0;

    
  
	for(int i=0; i<num_tml; ++i){
		string name;
		double x;
		double y;
		input >> str;	
    name=str;
		input >> str;
		input >> str;	
    x= double(stod(str));
		input >> str;  
    y= double(stod(str));
    name2tml[name] = new Terminal(name, x, y);

    name2index[name] = num_blk + i;
    cout << name <<" " << x << " " << y << endl;
         
    if (x > max_w)
    {
        max_w = x;
    }
    if (y > max_h)
    {
        max_h = y;
    }         

	}
    double x_ratio = new_chip_w / double(max_w);
    double y_ratio = new_chip_h / double(max_h);

    cout << "Resize " <<  total_area << " new_chip_w " <<new_chip_w << " new_chip_h " <<new_chip_h  << "  ratio " << x_ratio <<"  "<<  y_ratio << endl;
    // cout << "Initialize name2blk finished" << endl;
  
 
 
   for (const auto& entry : name2tml) {
      std::string name = entry.first;
      Terminal* t_temp = entry.second;
      t_temp->_x = t_temp->_x*x_ratio;  
      t_temp->_y = t_temp->_y*y_ratio;  
    }
 
  
 
    // cout << "Initialize name2tml finished" << endl;


	return;
}

void Floorplanner::initialize_edge_count(){
  for (const auto& net : net_list) {
        for (size_t i = 0; i < net->cell_list.size(); ++i) {
            for (size_t j = 0; j < net->cell_list.size(); ++j) {
                if (i !=j){
                    int nodeIndex1 = name2index[net->cell_list[i]];
                    int nodeIndex2 = name2index[net->cell_list[j]];
                    string new_name = nodeIndex1 + "_" + nodeIndex2;

                    if(edge_count.count(new_name)){
                        edge_count[new_name] += 1;
                    }
                    else{
                        edge_count[new_name] = 1;
                    }
                }
            }
        }
    }

    for (const auto& pair : edge_count) {
    // 检查当前值是否比最大值大
        if (pair.second > max_edge) {
            max_edge = pair.second;
        }
    }
    cout << "Max count "<< max_edge << endl;


    // cout << "Initialize node_list finished" << endl;
}


void Floorplanner::parse_net(const string& input_path){
    string str;
    fstream input;
    input.open(input_path, ios::in);
	input >> str;
	input >> str;	
    num_net = stod(str);
	for(int i=0; i<num_net; ++i){
		int eachNdegree;
		input >> str; 
		input >> str;	
        eachNdegree = stod(str);
        net_list.push_back(new Net());
	
		for(int j=0; j<eachNdegree; ++j){
			string name;
			input >> str;	
            name = str;
            net_list[i]->cell_list.push_back(name); 
		}
	}

    // cout << "Initialize net_list finished" << endl;
    
    initialize_edge_count();
	return;
}







void Floorplanner::initialize_node_list(){
    // after read and parse circuits, run this func
    // push_back node into node_list
    for(const auto& pair: name2blk){
        string name = pair.first;

		node_list.push_back(new Node(name));

    }


    for (const auto& net : net_list) {
        for (size_t i = 0; i < net->cell_list.size(); ++i) {
            for (size_t j = 0; j < net->cell_list.size(); ++j) {
                if (i !=j){
                    int nodeIndex1 = name2index[net->cell_list[i]];
                    int nodeIndex2 = name2index[net->cell_list[j]];
                    string new_name = nodeIndex1 + "_" + nodeIndex2;

                    if(edge_count.count(new_name)){
                        edge_count[new_name] += 1;
                    }
                    else{
                        edge_count[new_name] = 1;
                    }
                }
            }
        }
    }

    for (const auto& pair : edge_count) {
    // 检查当前值是否比最大值大
        if (pair.second > max_edge) {
            max_edge = pair.second;
        }
    }
    // cout << "Max count "<< max_edge << endl;


    // cout << "Initialize node_list finished" << endl;
}

void Floorplanner::partition(){
    for(auto i:name2blk){
        i.second->layer = rand() % num_layer;

    }
    for(auto i:name2tml){
        i.second->layer = rand() % num_layer;
    }
    // cout << "Initialize partition finished" << endl;
}





void Floorplanner::reset(int seed){
    // init partition, assign random layer index for each tml and blk
    partition();

    // nlk belonging to the same layer are assigned into one same vector
    vector<vector<Node*>> layer2nodes(num_layer);
    for(const auto& node:node_list){
        auto name = node->_name;
        int layer = name2blk[name]->layer;
        node->_left = nullptr;
        node->_right = nullptr;
        node->_prev = nullptr;
        layer2nodes[layer].push_back(node);
        
    }

    // shuffle layer2nodes
    //std::random_device rd;

    // 使用 std::default_random_engine 引擎和生成的随机种子
    std::default_random_engine rng(seed);

    // 使用 std::shuffle 打乱 node_list 中的数据
        // 对每一层的节点进行随机排序
    for (int layer = 0; layer < num_layer; ++layer) {
        std::shuffle(layer2nodes[layer].begin(), layer2nodes[layer].end(), rng);

        std::cout << "Shuffled order for layer " << layer << ": ";
        for (const auto& node : layer2nodes[layer]) {
            std::cout << node->_name << " ";
        }
        std::cout << std::endl;

    }

    // construct init tree for each layer
    for(int layer=0; layer<num_layer; ++layer){
        roots[layer] = layer2nodes[layer][0];
        for(uint i=1; i<layer2nodes[layer].size(); ++i){
            layer2nodes[layer][i-1]->_left = layer2nodes[layer][i];
            layer2nodes[layer][i]->_prev   = layer2nodes[layer][i-1];
        }
    }

    // calculate coordinates for all nodes
    get_all_nodes_coordinate();
}




void Floorplanner::initializeFrom_other_FP(const Floorplanner& other){

     name2tml = other.name2tml;
     node_list = other.node_list;
     name2Node = other.name2Node;
     name2blk = other.name2blk;
     roots = other.roots;

}


void Floorplanner::reset_wo_init_tree(){

    for(auto tml:name2tml){
        tml.second->layer = 0;
    }
    for (auto ptr : roots) {
        delete ptr; // 删除指针指向的对象
    }

    // 清空 roots 向量
    roots.clear();

    roots = vector<Node*>(num_layer, nullptr);
    node_list.clear();
    name2Node.clear();
    for(auto b:name2blk){
        b.second->_x = 0;
        b.second->_y = 0;
        b.second->on_FP = 0;
        b.second->layer = 0;
        b.second->left = 0;
        b.second->right = 0;
    }
}


int Floorplanner::coordinate(Node* node, int order){
    if(node!=nullptr){
        int layer_index = name2blk[node->_name]->layer;
        int caly=0;
        if(node->_prev==nullptr){
            name2blk[node->_name]->_x=0;
            name2blk[node->_name]->_y=0;      
            contour_lines[layer_index].push_back(ContourNode(0, name2blk[node->_name]->_w, name2blk[node->_name]->_h));
            y_max_each_layer[layer_index] = name2blk[node->_name]->_h;
            x_max_each_layer[layer_index] = name2blk[node->_name]->_w;

        }
        else if(node==node->_prev->_left){
            int x = name2blk[node->_prev->_name]->_x + name2blk[node->_prev->_name]->_w;
		    name2blk[node->_name]->_x = x;
            if(x + name2blk[node->_name]->_w > x_max_each_layer[layer_index]){
                x_max_each_layer[layer_index] = x + name2blk[node->_name]->_w;
            }
            caly = updcontour(node);
		    name2blk[node->_name]->_y=caly;
        }
        else if(node==node->_prev->_right){
            int x = name2blk[node->_prev->_name]->_x;
		    name2blk[node->_name]->_x = x;
            if(x + name2blk[node->_name]->_w > x_max_each_layer[layer_index]){
                x_max_each_layer[layer_index] = x + name2blk[node->_name]->_w;
            }
            caly = updcontour(node);
		    name2blk[node->_name]->_y = caly;
        }
        else{
            cout << node->_name + " is neither left or right" << endl;
        }
        //cout<<  "Info "   << node->_name << "__" << order << endl;
        order++;
        name2blk[node->_name]->order = order;
        order = coordinate(node->_left, order);
        order = coordinate(node->_right, order);
    } 

    return order;
}

double Floorplanner::SA_HPWL() {
    double totalLength = 0;
    for(uint i=0;  i<net_list.size(); ++i) {
        double minx = 2000000000, rightx = 0;
        double miny = 2000000000, upy = 0;
        int num = 0;
        for(uint j=0; j<net_list[i]->cell_list.size(); j++) {
            string name=net_list[i]->cell_list[j];
            // this is a block
            if (name2blk.count(name)) {
                Block* curblock = name2blk[name];
                double macroX = double(curblock->_x) + double(curblock->_w)/double(2);
                double macroY = double(curblock->_y) + double(curblock->_h)/double(2);
                minx = min(minx,macroX);
                rightx = max(rightx,macroX);
                miny = min(miny,macroY);
                upy = max(upy,macroY);
                num = num +1;
            }
            // this is a terminal
            else if(name2tml.count(name)) {
                Terminal* curTerminal = name2tml[name];
                double terminalX = curTerminal->_x;
                double terminalY = curTerminal->_y;
                minx = min(minx,terminalX);
                rightx = max(rightx,terminalX);
                miny = min(miny,terminalY);
                upy = max(upy,terminalY);
                num = num +1;
            }
            // error
            else{
                string error_msg = "In Floorplanner::HPWL(), cell " + name + " is not matched!\n" ;
                cout << error_msg;
                throw error_msg;
            }
        }

        totalLength += (rightx-minx) + (upy-miny);

    }
    return totalLength;
}


double Floorplanner::HPWL() {
    double totalLength = 0;
    for(uint i=0;  i<net_list.size(); ++i) {
        double minx = 2000000000, rightx = 0;
        double miny = 2000000000, upy = 0;
        int num = 0;
        for(uint j=0; j<net_list[i]->cell_list.size(); j++) {
            string name=net_list[i]->cell_list[j];
            // this is a block
            if (name2blk.count(name)) {
                if(name2blk[name]->on_FP==1){
                    Block* curblock = name2blk[name];
                    double macroX = double(curblock->_x) + double(curblock->_w)/double(2);
                    double macroY = double(curblock->_y) + double(curblock->_h)/double(2);
                    minx = min(minx,macroX);
                    rightx = max(rightx,macroX);
                    miny = min(miny,macroY);
                    upy = max(upy,macroY);
                    num = num +1;
                }
            }
            // this is a terminal
            else if(name2tml.count(name)) {
                Terminal* curTerminal = name2tml[name];
                double terminalX = curTerminal->_x;
                double terminalY = curTerminal->_y;
                minx = min(minx,terminalX);
                rightx = max(rightx,terminalX);
                miny = min(miny,terminalY);
                upy = max(upy,terminalY);
                num = num +1;
            }
            // error
            else{
                string error_msg = "In Floorplanner::HPWL(), cell " + name + " is not matched!\n" ;
                cout << error_msg;
                throw error_msg;
            }
        }
        if(num >= 2){
            totalLength += (rightx-minx) + (upy-miny);
        }
    }
    return totalLength;
}

double Floorplanner::calculate_outbound(){
    double x_error = max( double(get_x_max() -new_chip_w), 0.0);
    double y_error = max( double(get_y_max() -new_chip_h), 0.0);
    return x_error + y_error;
}


double Floorplanner::calculate_cost(){
//    double x_error = max( double(get_x_max() - new_chip_w), 0.0);
//    double y_error = max( double(get_y_max() - new_chip_h), 0.0);
//    if(x_error+y_error==0){
    double area = calculate_area();
    double hpwl = SA_HPWL();
    double feedthrough = calculate_feedthrough();
    return -(chip_w*chip_h)*5 / ( weight_area*area + weight_hpwl*hpwl + weight_feedthrough*feedthrough  );
//    }
//    else {
//        return (x_error+y_error)*10;
//    }
}

double Floorplanner::calculate_area(){
    return get_x_max() * get_y_max();
}


double Floorplanner::updcontour(Node* node){
    double y2, y1 = 0;
    double x1 = name2blk[node->_name]->_x;
    double x2 = x1+name2blk[node->_name]->_w;
    int layer_index = name2blk[node->_name]->layer;
	list<ContourNode>::iterator it = contour_lines[layer_index].begin();
	while(it != contour_lines[layer_index].end()){
		if(it->_x2 <= x1){
			++it;
		}
        else if(it->_x1 >= x2){
			break;
		}
        else{
			y1 = max(it->_y, y1);

			if(it->_x1>=x1 && it->_x2<=x2){
				it=contour_lines[layer_index].erase(it);
			}
            else if( it->_x1>=x1 && it->_x2>=x2 ){
				it->_x1=x2;
                continue;
			}
            else if( it->_x1<=x1 && it->_x2<=x2 ){
				it->_x2=x1;
                continue;
			}
            else{
				contour_lines[layer_index].insert(it,ContourNode(it->_x1,x1,it->_y));
				it->_x1=x2;
                continue;
			}
		}
	}
	y2 = y1+name2blk[node->_name]->_h;
	contour_lines[layer_index].insert(it,ContourNode(x1,x2,y2));
    if(y2 > y_max_each_layer[layer_index]) y_max_each_layer[layer_index] = y2;
    
	return y1;
}

using MyTuple = std::tuple<std::string, std::string, int, double, double>;

std::list<std::tuple<std::string, std::string, int, double, double>> Floorplanner::get_place_sequence(){

    std::list<MyTuple> dataList;

    map<int, string> orderToName;
    for (const auto& entry : name2blk) {
        orderToName[entry.second->order] = entry.first;
    }

    map<string, const Node*> nameToNodeMap;
    for (const auto& entry : node_list) {
        nameToNodeMap[entry->_name] = entry;
    }
    int temp_index = 0;
    for (const auto& entry : orderToName) {
        const std::string& blockName = entry.second;

        // 使用 nameToNodeMap 查找对应的 Node
        const Node* node = nameToNodeMap.at(blockName);

        if (temp_index == 0 ){

            dataList.push_back(std::make_tuple(node->_name, "None", 0, name2blk[roots[0]->_name]->_w, name2blk[roots[0]->_name]->_h));
        }else{
            if (node == node->_prev->_left)
            {
                dataList.push_back(std::make_tuple(node->_name, node->_prev->_name, 1, name2blk[node->_name]->_w, name2blk[node->_name]->_h));
            }else{

                dataList.push_back(std::make_tuple(node->_name, node->_prev->_name, 2, name2blk[node->_name]->_w, name2blk[node->_name]->_h));
            }
        }
        temp_index = temp_index+1;
        // 在这里使用 node
        // node 现在是按照 order 从小到大的顺序
    }


    return dataList;
  }
//    std::list<MyTuple> dataList;
//
//    std::stack<const Node*> nodeStack;
//    nodeStack.push(roots[0]);
//
//
//    dataList.push_back(std::make_tuple(roots[0]->_name, "None", 0, name2blk[roots[0]->_name]->_w, name2blk[roots[0]->_name]->_h));
//
//
//    while (!nodeStack.empty()) {
//        const Node* current = nodeStack.top();
//        nodeStack.pop();
//
//        // 处理当前节点
//        // 将右孩子先入栈，因为栈是先进后出的数据结构
//        // 将左孩子后入栈
//        if (current->_left != nullptr) {
//            nodeStack.push(current->_left);
//            dataList.push_back(std::make_tuple(current->_left->_name, current->_name, 1, name2blk[current->_left->_name]->_w, name2blk[current->_left->_name]->_h));
//        }
//        if (current->_right != nullptr) {
//            nodeStack.push(current->_right);
//            dataList.push_back(std::make_tuple(current->_right->_name, current->_name, 2, name2blk[current->_right->_name]->_w, name2blk[current->_right->_name]->_h));
//        }
//    }
//    return dataList;
//}


void Floorplanner::Insert_to_target_left_right_rotate(int step, string insert_name, string target_name, int left_or_right_rotate){

    Node* insert_node = new Node(insert_name);
    insert_node->_left = nullptr;
    insert_node->_right = nullptr;
    insert_node->_prev = nullptr;
    auto insert_blk = name2blk[insert_name];
    insert_blk->on_FP = 1;
    if(step==0)
    {
        node_list.push_back(insert_node);
       // cout << "add the first ????" << endl;
        roots[0] = node_list[0];
       // cout << 00000000000 <<  roots[0]->_name <<  endl;
//        for(auto root:roots){
//            cout << 3.1 <<  root->_name <<  endl;
//        }
        name2Node[insert_name] = insert_node;

    }else{

        auto target_blk = name2blk[target_name];
        node_list.push_back(insert_node);

        name2Node[insert_name] = insert_node;
        if (left_or_right_rotate == 0 || left_or_right_rotate == 1)
        {
            if (name2Node[target_name]->_left == nullptr)
            {
                 name2Node[target_name]->_left = insert_node;
                 insert_node->_prev = name2Node[target_name];
                 target_blk->left = 1;
            }else
            {
                 cout << name2Node[target_name]->_name + " has left, but want to insert one new left" << endl;
            }
        }
        else{
            if (name2Node[target_name]->_right == nullptr)
            {
                 name2Node[target_name]->_right = insert_node;
                 insert_node->_prev = name2Node[target_name];
                 target_blk->right = 1;
            }else
            {
                 cout << name2Node[target_name]->_name + " has right, but want to insert one new right" << endl;
            }
        }
    }
    if (left_or_right_rotate == 1 || left_or_right_rotate == 3)
    {
        insert_blk->rotate();
    }
}


tuple<int, int, int, int, Node*, Node*> Floorplanner::RL_delandins(int del, int ins, int left_or_right){
    // 0: del node is left child
    // 1: del node is right child
    int prev_lorr, child_lorr;
    Node* prevblk;
    Node* childblk;

    prevblk = node_list[del]->_prev;
    if(node_list[del]->_prev->_left == node_list[del]){
        prev_lorr = 0;
    }
    else{
        prev_lorr = 1;
    }

    // 0: del node has left child
    // 1: del node has right child
    // 2: del node has no child
    childblk = nullptr;
    // has left child
    if(node_list[del]->_left!=nullptr) {
        childblk=node_list[del]->_left;
        child_lorr=0;
    }
    // has right child
    else if(node_list[del]->_right!=nullptr) {
        childblk=node_list[del]->_right;
        child_lorr=1;
    }
    // has no child
    else{
        child_lorr=2;
        childblk = nullptr;
    }


    ////delete
    if(node_list[del]->_left != nullptr){
		node_list[del]->_left->_prev = node_list[del]->_prev;
		if(node_list[del]->_prev->_left == node_list[del]) node_list[del]->_prev->_left = node_list[del]->_left;
		else node_list[del]->_prev->_right = node_list[del]->_left;
		node_list[del]->_left = nullptr;
	}
    else if(node_list[del]->_right != nullptr){
		node_list[del]->_right->_prev = node_list[del]->_prev;
		if(node_list[del]->_prev->_left == node_list[del]) node_list[del]->_prev->_left = node_list[del]->_right;
		else node_list[del]->_prev->_right = node_list[del]->_right;
		node_list[del]->_right = nullptr;
	}
    else{
		if(node_list[del]->_prev->_left == node_list[del]) node_list[del]->_prev->_left = nullptr;
		else node_list[del]->_prev->_right = nullptr;
	}

    //// insert
    if(node_list[ins]->_left == nullptr && node_list[ins]->_right == nullptr){

        if (left_or_right == 0){
            node_list[ins]->_left = node_list[del];
        }
        else{
             node_list[ins] ->_right = node_list[del];
        }
        //if(rand()&1) node_list[ins]->_left = node_list[del];
        //else node_list[ins] ->_right = node_list[del];
    }
    else if(node_list[ins]->_left == nullptr){
        if(left_or_right == 0)
        {
            node_list[ins]->_left = node_list[del];
        }else {
            throw std::runtime_error("RL_delandins: insert to left requested but left_or_right != 0");
        }

    }
    else if(node_list[ins]->_right == nullptr){
        if(left_or_right == 1)
        {
            node_list[ins]->_right = node_list[del];
        }
        else {
            throw std::runtime_error("RL_delandins: insert to right requested but left_or_right != 1");
        }

    }
    node_list[del]->_prev = node_list[ins];

    name2blk[node_list[del]->_name]->layer = name2blk[node_list[ins]->_name]->layer;

    return {del, ins, prev_lorr, child_lorr, prevblk, childblk};

}




tuple<int, int, int, int, Node*, Node*> Floorplanner::delandins(int del, int ins){
    // 0: del node is left child
    // 1: del node is right child
    int prev_lorr, child_lorr;
    Node* prevblk;
    Node* childblk;

    prevblk = node_list[del]->_prev;
    if(node_list[del]->_prev->_left == node_list[del]){
        prev_lorr = 0;
    }
    else{
        prev_lorr = 1;
    }

    // 0: del node has left child
    // 1: del node has right child
    // 2: del node has no child
    childblk = nullptr;
    // has left child
    if(node_list[del]->_left!=nullptr) {
        childblk=node_list[del]->_left;
        child_lorr=0;
    }
    // has right child
    else if(node_list[del]->_right!=nullptr) {
        childblk=node_list[del]->_right;
        child_lorr=1;
    }
    // has no child
    else{
        child_lorr=2;
        childblk = nullptr;
    }


    ////delete
    if(node_list[del]->_left != nullptr){
		node_list[del]->_left->_prev = node_list[del]->_prev;
		if(node_list[del]->_prev->_left == node_list[del]) node_list[del]->_prev->_left = node_list[del]->_left;
		else node_list[del]->_prev->_right = node_list[del]->_left;
		node_list[del]->_left = nullptr;
	}
    else if(node_list[del]->_right != nullptr){
		node_list[del]->_right->_prev = node_list[del]->_prev;
		if(node_list[del]->_prev->_left == node_list[del]) node_list[del]->_prev->_left = node_list[del]->_right;
		else node_list[del]->_prev->_right = node_list[del]->_right;
		node_list[del]->_right = nullptr;
	}
    else{
		if(node_list[del]->_prev->_left == node_list[del]) node_list[del]->_prev->_left = nullptr;
		else node_list[del]->_prev->_right = nullptr;
	}

    //// insert
    if(node_list[ins]->_left == nullptr && node_list[ins]->_right == nullptr){
        if(rand()&1) node_list[ins]->_left = node_list[del];
        else node_list[ins] ->_right = node_list[del];
    }
    else if(node_list[ins]->_left == nullptr){
        node_list[ins]->_left = node_list[del];
    }
    else if(node_list[ins]->_right == nullptr){
        node_list[ins]->_right = node_list[del];
    }
    node_list[del]->_prev = node_list[ins];

    name2blk[node_list[del]->_name]->layer = name2blk[node_list[ins]->_name]->layer;

    return {del, ins, prev_lorr, child_lorr, prevblk, childblk};

}


void Floorplanner::swap(int node1, int node2){
    auto tmp_layer = name2blk[node_list[node1]->_name]->layer;
    auto tmp_name  = node_list[node1]->_name;

    name2blk[node_list[node1]->_name]->layer = name2blk[node_list[node2]->_name]->layer; // original node name1
    node_list[node1]->_name = node_list[node2]->_name; // set new node name for node 1

    name2blk[node_list[node2]->_name]->layer = tmp_layer; // original node name2
    node_list[node2]->_name = tmp_name;
}


void Floorplanner::revert_delandins(int del_node, int ins_node, int prev_lorr, int child_lorr, Node* prevblk, Node* childblk){
    if(node_list[ins_node]->_left == node_list[del_node])  node_list[ins_node]->_left=nullptr;
    else node_list[ins_node]->_right=nullptr;

    // del node is left child
    if(prev_lorr==0) {
        // del node has left child
        if(child_lorr==0){
            node_list[del_node]->_left=childblk;
            childblk->_prev=node_list[del_node];
        }
        // del node has right child
        else if(child_lorr==1){
            node_list[del_node]->_right=childblk;
            childblk->_prev=node_list[del_node];
        }
        // del node has no child
        else{}

        prevblk->_left=node_list[del_node];
    }

    // del node is right child
    else if(prev_lorr==1) {
        // del node has left child
        if(child_lorr==0){
            node_list[del_node]->_left=childblk;
            childblk->_prev=node_list[del_node];
        }
        // del node has right child
        else if(child_lorr==1){
            node_list[del_node]->_right=childblk;
            childblk->_prev=node_list[del_node];
        }
        // del node has no child
        else{}

        prevblk->_right=node_list[del_node];
    }

    // del node is neither left child nor right child
    else{
        cout << "del node is neither left child nor right child" << endl;
        throw "del node is neither left child nor right child";
    }


    node_list[del_node]->_prev=prevblk;
    name2blk[node_list[del_node]->_name]->layer = name2blk[prevblk->_name]->layer;

}


void Floorplanner::rotate(int node_idx){
    auto name = node_list[node_idx]->_name;
    auto blk  = name2blk[name];
    blk->rotate();
}

void Floorplanner::get_all_nodes_coordinate(){

    //cout << "???????????" <<  endl;
    for(auto root:roots){
        //cout<< "1581615" <<endl;
        if(root == nullptr) {
        cout<< root << " root null" <<endl;
        }
        else{
        //cout << 3.1 <<  root->_name <<  endl;
        int layer_index = name2blk[root->_name]->layer;
        x_max_each_layer[layer_index] = 0;
        y_max_each_layer[layer_index] = 0;
       // cout << 3.2 << endl;
        contour_lines[layer_index].clear();
       // cout << 3.3 << endl;
        coordinate(root);
        }
    }
}

string Floorplanner::write_report_string(double totaltime){
    std::ostringstream output;
 //   cout << 2.1 << endl;
    get_all_nodes_coordinate();
 //   cout << 2.2 << endl;
    output << fixed << "wirelength = " << (double)SA_HPWL() << endl;
    output << fixed << "area = " << (double)calculate_area() << endl;
    output << fixed << "feedthrough = " << (double)calculate_feedthrough() << endl;
    output << fixed << "x_max = " << (double)get_x_max() << endl;
    output << fixed << "y_max = " << (double)get_y_max() << endl;
    output << fixed << "cost = " << (double)calculate_cost() << endl;
    output << fixed << "totaltime = " << (double)totaltime << endl;
  //  cout << 2.3 << endl;
    for(size_t  i=0; i<node_list.size(); i++){
        auto node = node_list[i];
        auto block = name2blk[node->_name];

        output<<node->_name<<" "
            <<name2blk[node->_name]->_x<<" "<<name2blk[node->_name]->_y<<" "
            <<name2blk[node->_name]->_x+block->_w<<" "<<name2blk[node->_name]->_y+block->_h << " "
            <<name2blk[node->_name]->layer << " "
            <<name2blk[node->_name]->order << " "
            <<endl;
    }
    return output.str();
}

//void Floorplanner::write_report(double totaltime,const string& output_path){
void Floorplanner::write_report(double totaltime,const string& output_path){
    fstream output;
    output.open(output_path, ios::out);
 //   cout << 2.1 << endl;
    get_all_nodes_coordinate();
 //   cout << 2.2 << endl;
    output << fixed << "wirelength = " << (double)SA_HPWL() << endl;
    output << fixed << "area = " << (double)calculate_area() << endl;
    output << fixed << "feedthrough = " << (double)calculate_feedthrough() << endl;
    output << fixed << "x_max = " << (double)get_x_max() << endl;
    output << fixed << "y_max = " << (double)get_y_max() << endl;
    output << fixed << "cost = " << (double)calculate_cost() << endl;
    output << fixed << "totaltime = " << (double)totaltime << endl;
  //  cout << 2.3 << endl;
    for(size_t  i=0; i<node_list.size(); i++){
        auto node = node_list[i];
        auto block = name2blk[node->_name];

        output<<node->_name<<" "
            <<name2blk[node->_name]->_x<<" "<<name2blk[node->_name]->_y<<" "
            <<name2blk[node->_name]->_x+block->_w<<" "<<name2blk[node->_name]->_y+block->_h << " "
            <<name2blk[node->_name]->layer << " "
            <<name2blk[node->_name]->order << " " << name2blk[node->_name]->left << "  " <<name2blk[node->_name]->right
            <<endl;
    }
    output.close();
}
//
//void Floorplanner::write_report(){
//
//    get_all_nodes_coordinate();
//
//    cout << fixed << "wirelength = " << (double)HPWL() << endl;
//    cout << fixed << "area = " << (double)calculate_area() << endl;
//    cout << fixed << "feedthrough = " << (double)calculate_feedthrough() << endl;
//    cout << fixed << "x_max = " << (double)get_x_max() << endl;
//    cout << fixed << "y_max = " << (double)get_y_max() << endl;
//    cout << fixed << "cost = " << (double)calculate_cost() << endl;
//
//    for(int i=0; i<num_blk; i++){
//        auto node = node_list[i];
//        auto block = name2blk[node->_name];
//
//        cout<<node->_name<<" "
//            <<name2blk[node->_name]->_x<<" "<<name2blk[node->_name]->_y<<" "
//            <<block->_w<<" "<<block->_h << " "
//            <<name2blk[node->_name]->layer << " "
//            <<name2blk[node->_name]->order << " "
//            <<endl;
//    }
//}



map<string,double> Floorplanner::summary(int iteration_index){
    map<string,double> res;
    res["area"] = (double) calculate_area();
    res["hpwl"] = (double) HPWL();
    res["feedthrough"] = (double) calculate_feedthrough();
    res["iteration"] = (double) iteration_index;
    return res;
}


map<string, int> Floorplanner::get_name2idx(){
    // return a dict, block name to node index
    map<string, int> res;
    for(int i=0; i<num_blk; ++i){
        res[node_list[i]->_name] = i;
    }
    return res;
}


vector<vector<int>> Floorplanner::get_graph_src_dst_from_netlist(){
    // only contain block, not contain terminal
    vector<int> all_src, all_dst;
    map<string, int> name2idx = get_name2idx();

    for(Net* net:net_list){
        for(uint i=0; i<net->cell_list.size(); ++i){
            for(uint j=i+1; j<net->cell_list.size(); ++j){
                string name1 = net->cell_list[i];
                string name2 = net->cell_list[j];
                if(name2blk.count(name1) && name2blk.count(name2)){
                    all_src.push_back(name2idx[name1]);
                    all_dst.push_back(name2idx[name2]);
                }
            }
        }
    }
    return {all_src, all_dst};
}


vector<vector<int>> Floorplanner::get_graph_src_dst_from_tree(){
    vector<int> all_src, all_dst;
    map<string, int> name2idx = get_name2idx();

    for(int node_id=0; node_id<num_blk; ++node_id){
        Node* left = node_list[node_id]->_left;
        if(left != nullptr){
            int left_id = name2idx[left->_name];
            all_src.push_back(node_id);
            all_dst.push_back(left_id);
        }

        Node* right = node_list[node_id]->_right;
        if(right != nullptr){
            int right_id = name2idx[right->_name];
            all_src.push_back(node_id);
            all_dst.push_back(right_id);
        }
    }
    return {all_src, all_dst};
}


vector<vector<int>> Floorplanner::get_blk_feat(){
    // return feature for each block
    // for each block, add number of connected blocks as new feature
    map<string, int> name_to_num_neighbour_blk;
    for(Net* net: net_list){
        int num_blk_this_net = 0;
        for(string name:net->cell_list){
            if(name2blk.count(name)){
                num_blk_this_net++;
            }
        }
        for(string name:net->cell_list){
            if(name2blk.count(name)){
                name_to_num_neighbour_blk[name] = num_blk_this_net-1;
            }
        }
    }

    // for each block, add number of net as new feature
    map<string, int> name_to_num_net;
    for(Net* net: net_list){
        for(string name:net->cell_list){
            if(name2blk.count(name)){
                if(name_to_num_net.count(name)){
                    name_to_num_net[name] = 1;
                }
                else{
                    name_to_num_net[name] += 1;
                }
            }
        }
    }

    // chip_w, chip_h, x_max, y_max, num_blk
    // blk_id
    // x,y,z,w,h
    // number of connected blk, net
    vector<vector<int>> res;
    for(int node_id=0; node_id<num_blk; ++node_id){
        Block* b = name2blk[ node_list[node_id]->_name ];
        res.push_back({
            chip_w, chip_h, get_x_max(), get_y_max(), num_blk,
            node_id,
            b->_x, b->_y, b->layer, b->_w, b->_h, 
            name_to_num_neighbour_blk[node_list[node_id]->_name], name_to_num_net[node_list[node_id]->_name],
        });
    }
    return res;
}


vector<int> Floorplanner::get_all_root_ids(){
    // return a list of int
    vector<int> all_root_ids;
    for(int i=0; i<num_blk; ++i){
        if(node_list[i]->_prev == nullptr){
            all_root_ids.push_back(i);
        }
    }
    return all_root_ids;
}


vector<int> Floorplanner::get_nodes_with_two_children_ids(){
    // return a list of int
    vector<int> nodes_with_two_children_ids;
    for(int i=0; i<num_blk; ++i){
        if(node_list[i]->_left != nullptr && node_list[i]->_right != nullptr){
            nodes_with_two_children_ids.push_back(i);
        }
    }
    return nodes_with_two_children_ids;
}



tuple<Node*, int, Node*, Node*> Floorplanner::move_tree(int node_del, int node_ins){
    // you need to ensure that node_del and node_ins satisfy constraint
    Node* original_parent = node_list[node_del]->_prev;
    int is_original_left = 0;

    // node del is left child
    if(node_list[node_del]->_prev->_left == node_list[node_del]){
        node_list[node_del]->_prev->_left = nullptr;
        is_original_left = 1;
    }   
    // node del is right child
    else{
        node_list[node_del]->_prev->_right = nullptr;
        is_original_left = 0;
    }


    // set new parent for node del 
    node_list[node_del]->_prev = node_list[node_ins];


    // ins node's left is empty
    if(node_list[node_ins]->_left == nullptr){
        node_list[node_ins]->_left = node_list[node_del];
    }
    // ins node's right is empty
    else{
        node_list[node_ins]->_right = node_list[node_del];
    }

    // after move, layer should also be modified
    vector<int> subtree_node_indices = get_subtree_node_indices(node_del);
    for(int node_idx:subtree_node_indices){
        name2blk[node_list[node_idx]->_name]->layer = name2blk[node_list[node_ins]->_name]->layer;
    }


    return {original_parent, is_original_left, node_list[node_del], node_list[node_ins]};
}


void Floorplanner::revert_move_tree(Node* original_parent, int is_original_left, Node* node_del, Node* node_ins){
    // node_ins remove this child
    if(node_ins->_left == node_del){
        node_ins->_left = nullptr;
    }
    else{
        node_ins->_right = nullptr;
    }

    // node_del set original_parent
    node_del->_prev = original_parent;
    

    // original_parent set original child
    if(is_original_left){
        original_parent->_left = node_del;
    }
    else{
        original_parent->_right = node_del;
    }

    // set layer
    vector<int> subtree_node_indices = get_subtree_node_indices(get_name2idx()[node_del->_name]);
    for(int node_idx:subtree_node_indices){
        name2blk[node_list[node_idx]->_name]->layer = name2blk[original_parent->_name]->layer;
    }

    return;
}



vector<int> Floorplanner::get_subtree_node_indices(int node_idx){
    vector<int> subtree_node_indices;
    map<string,int> name2idx = get_name2idx();
    get_subtree_node_indices__(node_idx, subtree_node_indices, name2idx);
    return subtree_node_indices;
}


void Floorplanner::get_subtree_node_indices__(int node_idx, vector<int>& subtree_node_indices, map<string,int>&name2idx){
    subtree_node_indices.push_back(node_idx);
    Node* child;
    if(node_list[node_idx]->_left != nullptr){
        child = node_list[node_idx]->_left;
        get_subtree_node_indices__(name2idx[child->_name] , subtree_node_indices, name2idx);
    }
    if(node_list[node_idx]->_right != nullptr){
        child = node_list[node_idx]->_right;
        get_subtree_node_indices__(name2idx[child->_name] , subtree_node_indices, name2idx);
    }
}


int Floorplanner::is_adjacent(Block* b1, Block* b2){
    int res = 0;
    // A and B should be on the same layer
    if(b1->layer != b2->layer) return res;
    
    // AB
    if(b1->_x + b1->_w == b2->_x) res = 1;
    // BA
    if(b2->_x + b2->_w == b1->_x) res = 1;
    // A is above B
    if(b2->_y + b2->_h == b1->_y) res = 1;
    // B is above A
    if(b1->_y + b1->_h == b2->_y) res = 1;

    return res;
}


int Floorplanner::calculate_feedthrough(){
    int feedthrough = 0;
    for(Net* net:net_list){
        for(string name1: net->cell_list){
            for(string name2: net->cell_list){
                if(name2blk.count(name1) && name2blk.count(name2) && !is_adjacent(name2blk[name1], name2blk[name2])){
                    feedthrough++;
                }
            }
        }
    }
    return feedthrough / 2;
}



int Floorplanner::insert_node_to_tree(int node_src, int node_dst, int insert_to_right){
    // node_src should have no parent, no left child and no right child
    // node_dst should cannot have two children
    // cout << "insert start" << endl;

    Node* n1 = node_list[node_src];
    Node* n2 = node_list[node_dst];
    
    if(n1 == n2){
        cout << "n1 and n2 are both" << node_src << endl;
        return 0;
    }
    if(n1->_left != nullptr){
        cout << "n1->left != nullptr" << endl;
        return 0;
    }
    if(n1->_right != nullptr){
        cout << "n1->right != nullptr" << endl;
        return 0;
    }
    if(n1->_prev != nullptr){
        cout << "n1->prev != nullptr" << endl;
        return 0;
    }
    if(n2->_right != nullptr && insert_to_right){
        cout << "n2->right != nullptr, insert_to_right = " << insert_to_right  << endl;
        return 0;
    }
    if(n2->_left != nullptr && !insert_to_right){
        cout << "n2->left != nullptr, insert_to_right = " << insert_to_right  << endl;
        return 0;
    }

    n1->_prev = n2;
    if(insert_to_right) n2->_right = n1;
    else n2->_left = n1;
    // cout << "insert end" << endl;

    name2blk[ n1->_name ]->layer = name2blk[ n2->_name ]->layer;

    return 1;
}



void Floorplanner::set_roots(vector<int> new_roots){
    for(uint i=0; i<new_roots.size(); ++i){
        roots[i] = node_list[new_roots[i]];
    }
}



