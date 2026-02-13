#ifndef RESOURCES_HANDLER
#define RESOURCES_HANDLER

#include "datatypes.h"
#include "fileshandler.h"

#define makePath(...) \
    makePath_( (int)(sizeof((char*[]){__VA_ARGS__}) / sizeof(char*)), \
               (char*[]){__VA_ARGS__} )



typedef struct Resources_
{
    char *path;
    void *return_data;
    struct Resources_(*getRoute)(struct Resources_* parent);
    struct Resources_(*getResource)(struct Resources_ *parent,char * resName,int isFile);
    char * (*toString)(struct Resources_ * parent);
}Resources;



Resources _Resources(Resources *parent,char * name);
Resources getResource(Resources *parent,char * resName,int isFile);
char *toString(Resources *parent);
char *makePath_(int count,char *paths[]);
void destroyResources(Resources * res);
#endif