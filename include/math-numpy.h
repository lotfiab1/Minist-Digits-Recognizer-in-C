#ifndef MATH_NUMPY
#define MATH_NUMPY

#include "datatypes.h"
#include <math.h>
#include "cblas.h"
#include <time.h>
typedef struct 
{
    Matrix *X;
    Matrix *Y;
    long num_batches;
}Batch;
long randInteger(long start,long end);
float relu(float a);
float deriv_relu(float z);
void softmax(Matrix z, Matrix *out);
float rand_uniform();
void destroyBatch(Batch *batch);
void dot(Matrix A,Matrix B,int TronsposeA,int TronsposeB,Matrix scaller,Matrix *res1,Matrix *res2,float alpha1,float alpha2,float (*Operation)(float a),const char * logs);
Batch batchSamples(Matrix X,Matrix Y,long batch_size,long num_samples);
void shuffleBatch(Batch *batch);
#endif