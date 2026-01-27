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

