#ifndef DATATYPES_IMPLEMENTATION
#define DATATYPES_IMPLEMENTATION


#include <stdio.h>
#include <string.h>
#include <stdlib.h>


#define BUFF_SIZE 4096

#define MemoryError() printf("[!] Inssuficient Memory\n")



typedef struct {
    long rows,cols;
    float *data;
}Matrix;
Matrix makeMatrix(long rows,long cols);
Matrix appendMatrix(Matrix dst,Matrix src);

void destroyMatrix(Matrix *m);
void printMatrix(Matrix m);
#endif