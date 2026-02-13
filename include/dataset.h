#ifndef DATASET_IMPLEMENTATION
#define DATASET_IMPLEMENTATION

#include "datatypes.h"
#include "fileshandler.h"
#include "math-numpy.h"


#define WIDTH 28
#define HEIGHT 28


typedef struct 
{
    Matrix images;
    Matrix labels;
}Dataset;


Dataset parseCsv(void *data,char delim);
void split_set(Dataset *main_set,Dataset *train_set,Dataset *test_set,float test_size);
#endif