#ifndef LLM_DIGITS_MINIST_MODEL
#define LLM_DIGITS_MINIST_MODEL

#include "datatypes.h"
#include "math-numpy.h"

typedef struct{
    Matrix w1,w2,b1,b2;
}Params;

typedef struct{
    Matrix z1,z2,a1,a2;
}Layers;

typedef struct {
    Matrix dz1,dz2;
}Gradients;

typedef struct {
    long input_layer,hidden_layer,output_layer;
    Params trained_params;

}llmModele;
void initParams(Params *p, long in,long hid, long out);
void freeParams(Params *p);
llmModele train(Matrix X,Matrix Y,long num_iteration,long batch_size,float lr,long input_layer,long hidden_layer,long output_layer);

void predict(Matrix X,Matrix Y,Params p,Layers *a,Matrix * one_hot_y);

void forward(Matrix *X,Params *P,Layers *L);
void backward(Matrix X,Params P,Matrix Y,Layers L,Gradients *G,float lr);
void updateParams(Matrix X,Layers L,Gradients G,Params *P,float lr);
void saveModele(llmModele model,const char * modele_name);
llmModele loadModel(const char *filename);


#endif