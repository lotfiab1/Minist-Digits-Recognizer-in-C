#include "dataset.h"
#include "resourceshandler.h"
#include "llm-digits-minist-model.h"

#define IN_L 784
#define HID_L 512
#define OUT_L 10

int main(int argc,char * argv[]){
    //Train 
    const char * filename = "modeles/"LLM_NAME LLM_VERSION".pd"; //to save modele
    printf("filename : %s\n",filename);
    return 0;
    Resources res = _Resources(NULL,"./res");
    if(!res.path) return 1;
    Resources dataset_csv = res.getResource(&res,"dataset.csv",1);
    if(!dataset_csv.return_data){
        printf("[!] failed to get resource %s\n",dataset_csv.path);
        destroyResources(&res);
        return 1;
    }
    Dataset train_set,test_set;
    Dataset dataset = parseCsv(dataset_csv.return_data,',');

    split_set(&dataset,&train_set,&test_set,0.2);
    printf("%s Loaded Succefully\n",dataset_csv.path);
    printf("------------------- INFO ----------------\n");
    printf("IMAGE : num_samples = %d num_pixels = %d\n",dataset.images.rows,dataset.images.cols);
    printf("Labels : num_samples = %d \n",dataset.labels.rows);
    printf("Train set size : %ld\n",train_set.images.rows);
    printf("Test set size : %ld\n",test_set.labels.rows);
    printf("-----------------------------------------\n");
    llmModele model = train(train_set.images,train_set.labels,100,256,0.1,IN_L,HID_L,OUT_L);
    
    //test

    printf("TEST:\n\n");
    predict(test_set.images,test_set.labels,model.trained_params,NULL,NULL);
    
    
    //save modele
    saveModele(model,filename);


    freeParams(&model.trained_params);
    destroyMatrix(&dataset.labels);
    destroyMatrix(&dataset.images);
    destroyResources(&res);
    destroyResources(&dataset_csv);



    return 0;
}















