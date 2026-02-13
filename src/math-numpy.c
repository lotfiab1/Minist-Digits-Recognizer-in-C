#include "math-numpy.h"

unsigned char isInit = 0;

long randInteger(long start,long end){
    if(!isInit) {
        srand(time(NULL));
        isInit = 1;
    }
    if(start == end) return start;
    return start + ((long)rand() % (end - start));
}


void softmax(Matrix z, Matrix *out) {
    //z(N,M)
    for (long i = 0; i < z.rows; i++){
        float max = z.data[0];

        for (long j = 1; j < z.cols; j++)
        {
            if (z.data[i * z.cols + j] > max) max = z.data[i * z.cols + j];
        }

        float sum = 0.0f;

        for(long j = 0; j < z.cols;j++){
            float val = expf(z.data[i * z.cols + j] - max);
            out->data[i * z.cols + j] = val;
            sum += val;
        }

        for (long j = 0; j < z.cols; j++)
        {
            out->data[i * z.cols + j] /= sum;
        }
        
    }
    return;
}

float relu(float a){
    return (a < 0)?0:a;
}

float deriv_relu(float z){
    return (z > 0)?1:0;
}

float rand_uniform(){
    if(!isInit){
        srand(time(NULL));
        isInit = 1;
    }
    return (float)rand() / RAND_MAX;
}
void *_lmemcpy(void * dest,void *src,size_t size){
    return memcpy((dest)?dest:malloc(size),src,size);
}

Batch batchSamples(Matrix X,Matrix Y,long batch_size,long num_samples)

{
    //We assume that X(n,784) Y(n,1)
    long _r = num_samples % batch_size;
    long num_batches = num_samples / batch_size;
    if(_r > 0) num_batches++;
    Batch batch = {
        .X = (Matrix *)calloc(num_batches,sizeof(Matrix)),
        .Y = (Matrix *)calloc(num_batches,sizeof(Matrix)),
        .num_batches = num_batches
    };
    if(!batch.X || !batch.Y) return (Batch){0};
    long _Copiedsamples = 0;
    long _Copysamples;
    for (long i = 0; i < num_batches; i++)
    {
        _Copysamples = (_Copiedsamples + batch_size > num_samples)?(num_samples - _Copiedsamples):batch_size;
        if(_Copysamples <= 0){
            batch.X = realloc(batch.X,i * sizeof(Matrix));
            batch.Y = realloc(batch.Y,i * sizeof(Matrix));
            batch.num_batches = i;
            break;
        } 
        batch.X[i].data = _lmemcpy(NULL,&X.data[_Copiedsamples * X.cols],_Copysamples * X.cols * sizeof(float));
        batch.Y[i].data = _lmemcpy(NULL,&Y.data[_Copiedsamples],_Copysamples * sizeof(float));
        
        batch.X[i].cols = X.cols;
        batch.Y[i].cols = batch.X[i].rows = _Copysamples;
        batch.Y[i].rows = 1;
        
        _Copiedsamples += _Copysamples;
    }
    return batch;
}
void shuffleBatch(Batch *batch)
{
    if(!batch || !batch->X || !batch->Y) exit(1);
    long cols = batch->X->cols;
    long rows;
    long s = cols * sizeof(float);
    float *buff = malloc(s);
    for (long i = 0; i < batch->num_batches; i++)
    {
        rows = batch->X[i].rows;
        for (long j = 0; j < rows; j++)
        {
            long idx1 = j * cols;
            long rnd = randInteger(j,rows);
            long idx2 = rnd * cols;
            memcpy(buff,&batch->X[i].data[idx1],s);
            memcpy(&batch->X[i].data[idx1],&batch->X[i].data[idx2],s);
            memcpy(&batch->X[i].data[idx2],buff,s);

            float val = batch->Y[i].data[j];
            batch->Y[i].data[j] = batch->Y[i].data[rnd];
            batch->Y[i].data[rnd] = val;

            
        }
        
    }

    free(buff);
    
}

void destroyBatch(Batch *batch)
{
    if(!batch) return;

    for (long i = 0; i < batch->num_batches; i++)
    {
        destroyMatrix(&batch->X[i]);
        destroyMatrix(&batch->Y[i]);

    }
    return;
    
}
void dot(Matrix A,Matrix B,int TronsposeA,int TronsposeB,Matrix scaller,Matrix *res1,Matrix *res2,float alpha1,float alpha2,float (*Operation)(float a),const char * logs){
    
    
    long rowA = (TronsposeA)?A.cols:A.rows;
    long colA = (TronsposeA)?A.rows:A.cols;
    long rowB = (TronsposeB)?B.cols:B.rows;
    long colB = (TronsposeB)?B.rows:B.cols;


    if(colA != rowB){
        printf("[!] Multiplication of two matrix with different shape (%ld,%ld) (%ld,%ld)\n",A.rows,A.cols,rowB,colB);
        printf("[LOGS] : Failed at %s\n ",logs);
        exit(1);
    }
    if(!res1->data){
            *res1 = makeMatrix(rowA,colB);
    }
    if(res2){
        if(!res2->data){
            makeMatrix(rowA,colB);
        } 
    }

    cblas_sgemm(
        CblasRowMajor,
        TronsposeA?CblasTrans:CblasNoTrans,
        TronsposeB?CblasTrans:CblasNoTrans,
        rowA,        
        colB,       
        colA,        
        alpha1,
        A.data, A.cols,    
        B.data, B.cols,     
        alpha2,
        res1->data, res1->cols  
    );
    if(res2 && Operation){
        for (long i = 0; i < res1->rows; i++)
        {
            float b = 0;
            if(scaller.data) b = res1->data[i];
            
            for (long j = 0; j < res1->cols; j++)
            {
                res1->data[i * colB + j] += b;
                res2->data[i * colB + j] = Operation(res1->data[i * colB + j]);
            }
            
        }
    }

    return;

}
