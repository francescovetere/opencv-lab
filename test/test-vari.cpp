#include <iostream>
#include <fstream>
#include <iterator>

int main() {
    std::istream_iterator<std::string> i(std::cin);
    std::istream_iterator<std::string> i_end;
    
    std::ofstream f1;
    f1.open("test.txt");
    
    std::ostream_iterator<std::string> out(f1, " ");

    std::copy(i, i_end, out);

    return 0;
}