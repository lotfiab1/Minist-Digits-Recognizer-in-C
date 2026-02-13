#include "llm-digits-minist-model.h"

void initParams(Params *p, long in,long hid, long out)
{
    double limit1 = sqrt(6.0 / (in + hid));
    double limit2 = sqrt(6.0 / (hid + out));
    p->w1 = makeMatrix(hid,in);
    p->b1 = makeMatrix(hid,1);
    p->w2 = makeMatrix(out,hid);
    p->b2 = makeMatrix(out,1);
    for (long i = 0; i < hid; ++i) {
        long row = i * in; 
        //init w1
        for (long j = 0; j < in; j++)
        {
            p->w1.data[row + j] = (2.0 * rand_uniform() - 1.0) *limit1;
        }
        
        //init w2
        for (long j = 0; j < out; j++)
        {
            p->w2.data[j * hid + i] = (2.0 * rand_uniform() - 1.0) *limit2;
        }
        
    }

    return;
}
void initGradients(Gradients *g,long numSamples,long in,long hid,long out){
    g->dz1 = makeMatrix(numSamples,hid);
    g->dz2 = makeMatrix(numSamples,out);
}
void initActivations(Layers *l,long numSamples,long in,long hid,long out){
    l->a1 = makeMatrix(numSamples,hid);
    l->a2 = makeMatrix(numSamples,out);
    l->z1 = makeMatrix(numSamples,hid);
    l->z2 = makeMatrix(numSamples,out);
}
void destroyGradient(Gradients *g){
    destroyMatrix(&g->dz1);
    destroyMatrix(&g->dz2);
}
void freeBatchGradients(Gradients *g,long num_batch){
    
    for(long i = 0;i < num_batch;i++){
        destroyGradient(&g[i]);
    }
}
void destroyActivation(Layers *l){
    destroyMatrix(&l->a1);
    destroyMatrix(&l->a2); 
    destroyMatrix(&l->z1);
    destroyMatrix(&l->z2);
}
void freeBatchActivations(Layers *l,long num_batch){
    for(long i = 0; i < num_batch;i++){
        destroyActivation(&l[i]);
    }
}
void freeParams(Params *p){
    if(!p) return;

    destroyMatrix(&p->w1);
    destroyMatrix(&p->b1);
    destroyMatrix(&p->w2);
    destroyMatrix(&p->b2);
}

float lossEntropy(Matrix a, Matrix y) {
    float loss = 0.0f;
    float eps = 1e-7f; 
    long num_samples = a.rows;
    long num_classes = a.cols;

    for (long i = 0; i < num_classes * num_samples; i++) {
        if (y.data[i] > 0.5f) {
            float prob = a.data[i];
            
            if (prob < eps) prob = eps;
            if (prob > 1.0f - eps) prob = 1.0f - eps;
            
            loss += logf(prob);
        }
    }
    
    return -loss / (float)num_samples;
}

void oneHotEncoding(Matrix a,Matrix y,Matrix *one_hot_y){
    if(!one_hot_y){
        one_hot_y = malloc(sizeof(Matrix));
        one_hot_y->data = makeMatrix(a.rows,a.cols).data;
        one_hot_y->cols = a.cols;
        one_hot_y->rows = a.rows;
        if(!one_hot_y->data) return;
    }

    memcpy(one_hot_y->data,a.data,a.cols * a.rows * sizeof(float));

    for (long i = 0; i < a.rows; i++)
    {
        long label = (int)y.data[i];
        one_hot_y->data[i * a.cols + label] -= 1.0f; 
        
    }
    
}
void predict(Matrix X,Matrix Y,Params p,Layers *a,Matrix * one_hot_y){
    int local_alloc = 0;
    Layers local_layer;
    int layer_alloc = 0;
    if(!a){
        long in_l = X.rows;
        long hid_l = p.b1.rows;
        long out_l = p.b2.rows;
        initActivations(&local_layer,X.rows,in_l,hid_l,out_l);
        layer_alloc = 1;
    }
    else local_layer = *a;
    
    forward(&X,&p,&local_layer);
    Matrix one;
    if(!one_hot_y){
        one = makeMatrix(local_layer.a2.rows,local_layer.a2.cols);
        local_alloc = 1;
    }
    else one = * one_hot_y;
    oneHotEncoding(local_layer.a2,Y,&one);
    float accuracy;
    long correct = 0;
    int best = 0;
    for (long i = 0; i < local_layer.a2.rows; i++)
    {
        int label = (int)Y.data[i];
        best = 0;
        for (long j = 1; j < local_layer.a2.cols; j++)
        {
            if(local_layer.a2.data[i * local_layer.a2.cols + j] > local_layer.a2.data[i * local_layer.a2.cols + best]) best = j; 
        }

        if(best == label) correct++;
        
    }

    accuracy = (float)correct / (float)X.rows;
    
    float loss = lossEntropy(local_layer.a2,one);


    
    //logs

    printf("Total samples : %ld\n",X.rows);
    printf("Correct Predictions : %ld\n",correct);
    printf("Accuracy : %.2f\n",accuracy);
    printf("Loss : %f\n",loss);

    if(local_alloc) destroyMatrix(&one);
    if(layer_alloc) destroyActivation(&local_layer);



    return;
}

void backward(Matrix X,Params P,Matrix Y,Layers L,Gradients *G,float lr)
{
    oneHotEncoding(L.a2,Y,&G->dz2);
    // W2.T.dot(dz2) W2(10,512) dz2(N,10)
    dot(G->dz2,P.w2,0,0,(Matrix){0},&G->dz1,0,1,0,NULL,"dz2.dot(W2)");
    for (long i = 0; i < G->dz1.rows * G->dz1.cols; i++)
    {
       
        G->dz1.data[i] *= deriv_relu(L.z1.data[i]);
        
    }

    return;
}

void forward(Matrix *X,Params *p,Layers *l){
    dot(*X,p->w1,0,1,p->b1,&l->z1,&l->a1,1,0,relu,"X.dot(w1) + b1");
    dot(l->a1,p->w2,0,1,p->b2,&l->z2,NULL,1,0,NULL,"a1.dot(w2) + b2");
    softmax(l->z2,&l->a2);
}
void updateParams(Matrix X,Layers L,Gradients G,Params *P,float lr){
    // dz2.dot(a1.T) dz2(N,10) a1(N,512)
    float inv = 1 / (float)X.rows;
    float alpha = -lr * inv;
    dot(G.dz2,L.a1,1,0,(Matrix){0},&P->w2,0,alpha,1.0,0,"dz2.dot(a1.T) dz2(N,10) a1(N,128)");
    
    float *ones = malloc(sizeof(float) * X.rows);

    memset(ones,1,sizeof(float) * X.rows);

    cblas_sgemv(CblasRowMajor, CblasTrans, 
                G.dz2.rows, G.dz2.cols, 
                alpha, G.dz2.data, G.dz2.cols, 
                ones, 1, 
                1.0, P->b2.data, 1);


    dot(G.dz1,X,1,0,(Matrix){0},&P->w1,0,alpha,1.0,0,"dz1.dot(X) dz1(N,128) X(N,784)");
    //w1(512,784)
    // dz1.dot(X) dz1(N,512) X(N,784)
    cblas_sgemv(CblasRowMajor, CblasTrans, 
                G.dz1.rows, G.dz1.cols, 
                alpha, G.dz1.data, G.dz1.cols, 
                ones, 1, 
                1.0, P->b1.data, 1);


    free(ones);
    return;
}


llmModele train(Matrix X,Matrix Y,long num_iteration,long batch_size,float lr,long input_layer,long hidden_layer,long output_layer)
{
    Batch mainB = {.X = (Matrix[1]){X},.Y = (Matrix[1]){Y},.num_batches = 1};
    shuffleBatch(&mainB);
    Batch batch = batchSamples(X,Y,batch_size,X.rows);
    if(!batch.num_batches) return (llmModele){0};
    Layers a;
    Gradients batch_g[batch.num_batches];
    Layers batch_a[batch.num_batches];
    Matrix one_hot_y = makeMatrix(X.rows,output_layer);
    Params p;

    initParams(&p,input_layer,hidden_layer,output_layer);
    initActivations(&a,X.rows,input_layer,hidden_layer,output_layer);
    
    for (long i = 0; i < batch.num_batches; i++)
    {
        initGradients(
        &batch_g[i],
        batch.X[i].rows,
        input_layer,
        hidden_layer,
        output_layer
        );
        initActivations(
        &batch_a[i],
        batch.X[i].rows,
        input_layer,
        hidden_layer,
        output_layer
        );

    }
    //training

    for (long i = 0; i < num_iteration; i++)
    {
        for (long j = 0; j < batch.num_batches; j++)
        {
            forward(&batch.X[j],&p,&batch_a[j]);
            backward(batch.X[j],p,batch.Y[j],batch_a[j],&batch_g[j],lr);
            updateParams(batch.X[j],batch_a[j],batch_g[j],&p,lr);
        }
        if(i % 50 == 0 && i > 0){
            lr *= 0.08; 
            printf("lr = %f\n",lr);
        }
        if(i % 10 == 0){
            printf("Train test iter %ld:\n",i);
            predict(X,Y,p,&a,&one_hot_y);
        }


        shuffleBatch(&batch);
        
    }
    

    //end training

    freeBatchActivations(batch_a,batch.num_batches);
    freeBatchGradients(batch_g,batch.num_batches);
    destroyActivation(&a);
    destroyBatch(&batch);
    destroyMatrix(&one_hot_y);
    return (llmModele){
        .input_layer = input_layer,
        .hidden_layer = hidden_layer,
        .output_layer = output_layer,
        .trained_params = p
    };
    
}
void saveMatrix(Matrix m,FILE *f){
    fwrite(&m.rows,sizeof(long),1,f);
    fwrite(&m.cols,sizeof(long),1,f);
    fwrite(m.data,sizeof(float),m.cols * m.rows,f);
    return;
}
void saveParams(Params p,FILE * f){
    saveMatrix(p.w1,f);
    saveMatrix(p.w2,f);
    saveMatrix(p.b1,f);
    saveMatrix(p.b2,f);

    return;
}
void saveModele(llmModele model,const char * modele_name){
    FILE *f = fopen(modele_name,"wb");
    if(!f){
        printf("[!] Failed save modele\n");
        return;
    }
    fwrite(&model.input_layer,sizeof(long),1,f);
    fwrite(&model.hidden_layer,sizeof(long),1,f);
    fwrite(&model.output_layer,sizeof(long),1,f);
    saveParams(model.trained_params,f);
    
    fclose(f);

    return;

}
void loadMatrix(FILE *f, Matrix *m)
{
    fread(&m->rows, sizeof(long), 1, f);
    fread(&m->cols, sizeof(long), 1, f);

    m->data = malloc(sizeof(float) * m->rows * m->cols);

    fread(m->data, sizeof(float), m->rows * m->cols, f);
}
void loadParams(FILE *f,Params *p){
    loadMatrix(f,&p->w1);
    loadMatrix(f,&p->w2);
    loadMatrix(f,&p->b1);
    loadMatrix(f,&p->b2);
}
llmModele loadModel(const char *filename)
{
    FILE *f = fopen(filename, "rb");
    if(!f){
        printf("Failed load modele %s\n",filename);
        return (llmModele){0};
    }

    llmModele m;

    fread(&m.input_layer, sizeof(long), 1, f);
    fread(&m.hidden_layer, sizeof(long), 1, f);
    fread(&m.output_layer, sizeof(long), 1, f);

    loadParams(f,&m.trained_params);

    fclose(f);
    return m;
}