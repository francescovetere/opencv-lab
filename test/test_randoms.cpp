#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>

int main() {
    std::vector<int> points0 {8, 2, 3, 2, 5, 2, 9};
    std::vector<int> points1 {3, 1, 6, 5, 2, 2, 8};
    std::vector<int> sample0, sample1;
    int sample_size = 4;

    srand(time(NULL));

    // Seleziono i 4 match casuali
    for(int i = 0; i < sample_size; ++i) {
      // Scelgo un indice random, col quale prelevo i punti dai 2 array di punti
      // I punti però non vengono inseriti se risultano duplicati, altrimenti chiaramente l'omografia non andrebbe a buon fine

      int index;
      int val0, val1;

      do {
        index = rand() % points1.size();

        val0 = points0[index];
        val1 = points1[index];
        
        if(std::find(sample0.begin(), sample0.end(), val0) != sample0.end())
            std::cout << val0 << " already in sample0. Repeating." << std::endl;
        else
            std::cout << val0 << " is ok." << std::endl;

        if(std::find(sample1.begin(), sample1.end(), val1) != sample1.end())
            std::cout << val1 << " already in sample1. Repeating." << std::endl;
        else
            std::cout << val1 << " is ok." << std::endl;
      
      } while(
             std::find(sample0.begin(), sample0.end(), val0) != sample0.end() ||
             std::find(sample1.begin(), sample1.end(), val1) != sample1.end()
            );

      sample0.push_back(val0);
      sample1.push_back(val1);
    }

    std::cout << "sample 0: [";
    for(int i = 0; i < sample0.size(); ++i) {
      std::cout << sample0[i] << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "sample 1: [";
    for(int i = 0; i < sample1.size(); ++i) {
      std::cout << sample1[i] << " ";
    }
    std::cout << "]" << std::endl;
}