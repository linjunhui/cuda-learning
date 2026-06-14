#include "utils.hpp"
#include <cstdlib>
#include <math.h>
#include <random>

void initMatrix(float *data, int size, int max, int min, int seed) {
    srand(seed);
    for(int i = 0; i < size; i++) {
        float scale = float(rand()) / RAND_MAX;
        data[i] = scale * (max - min) + min;
        
    }
}