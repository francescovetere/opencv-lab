#include <stdio.h>

int main() {
    float data[2] = {3.14, 1.44};
    float* p1 = data;
    float* p2 = &data[0];
    float* p3 = (float*) &data;

    printf("data: %p\n&data[0]: %p\n&data: %p\n", p1, p2, p3);

    return 0;
}