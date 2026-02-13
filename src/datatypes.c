#include "datatypes.h"

Matrix makeMatrix(long rows,long cols)
{
    return (Matrix){
        .data = (float *) calloc(rows * cols,sizeof(float)),
        .rows = rows,
        .cols = cols
    };
}

Matrix appendMatrix(Matrix dst,Matrix src){
    if(!dst.data && !src.data) return (Matrix){0};
    Matrix dst2 = {
        .data = realloc(dst.data,(src.cols)*(src.rows + dst.rows)*sizeof(float)),
        .rows = src.rows + dst.rows,
        .cols = src.cols
    };
    if(!dst2.data) return (Matrix){0};
    memcpy(&dst2.data[dst.cols * dst.rows],src.data,src.cols * sizeof(float));
    return dst2;
}
void destroyMatrix(Matrix *m)
{
    if(!m) return;
    if(!m->data) return;
    free(m->data);
    m->data = 0;
    m->rows = m->cols = 0;
}
void printMatrix(Matrix m)
{
    for (long i = 0; i < m.rows; i++)
    {
        for (long j = 0; j < m.cols; j++)
        {
            printf("%.2f ",m.data[i * m.cols + j]);
        }
        printf("\n");
        
    }
    
    return;
}