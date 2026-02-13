#include "dataset.h"
static inline int __atoi(const char **p)
{
    int v = 0;
    while (*(*p) >= '0') {
        v = v * 10 + (*(*p) - '0');
        (*p)++;
    }
    if (**p == ',') (*p)++;
    return v;
}
void split_set(Dataset *main_set,Dataset *train_set,Dataset *test_set,float test_size){
    if(!main_set || !train_set || !test_set ) return;
    Batch b = {
        .X = (Matrix [1]){main_set->images},
        .Y = (Matrix [1]){main_set->labels},
        .num_batches = 1
    };

    shuffleBatch(&b);

    long num_samples = main_set->images.rows;
    long num_test_samples  = num_samples * test_size;
    long num_train_samples = num_samples - num_test_samples;

    train_set->images = (Matrix){
        .data = main_set->images.data,
        .cols = main_set->images.cols,
        .rows = num_train_samples
    };
    train_set->labels = (Matrix){
        .data = main_set->labels.data,
        .cols = main_set->labels.cols,
        .rows = num_train_samples
    };
    long idx = num_train_samples * main_set->images.cols;
    test_set->images = (Matrix){
        .data = &main_set->images.data[idx],
        .cols = main_set->images.cols,
        .rows = num_test_samples
    };
    test_set->labels = (Matrix){
        .data = &main_set->labels.data[num_train_samples],
        .cols = main_set->labels.cols,
        .rows = num_test_samples
    };


    return;
}
long countIemPerLine(char * data,char delim){
    if(!data) return 0;
    long count = 0;
    for(char *p = data;*p && *p != '\n';p++)
    {
        if(*p == delim) count++;
    }

    return ++count;

}
Dataset parseCsv(void *data,char delim)
{
    if(!data) return (Dataset){0};
    long lines = count_line(data);
    long item_per_line = countIemPerLine(data,delim) - 1; // -1 for label
    char *start = data;
    Matrix pixels = makeMatrix(lines,item_per_line);
    Matrix labels = makeMatrix(lines,1);
    if(!pixels.data || !labels.data){
        MemoryError();
        destroyMatrix(&pixels);
        destroyMatrix(&labels);
        return (Dataset){0};
    }
    
    
    int row = 0;
    while (*start) {
        labels.data[row] = (float)__atoi((const char**)&start);
        int base = row * pixels.cols;
        for (int i = 0; i < item_per_line; i++) {
            pixels.data[base + i] =
                __atoi((const char**)&start) * (1.0f / 255.0f);
        }

        while (start && *start != '\n') start++;
        start++;
        row++;
    }

    if(row < lines){
        pixels.data = realloc(pixels.data,row * pixels.cols * sizeof(float));
        labels.data = realloc(labels.data,row * sizeof(float));
        pixels.rows = labels.cols = row;
    }
    return (Dataset){ 
        .labels = labels,
        .images = pixels };
}