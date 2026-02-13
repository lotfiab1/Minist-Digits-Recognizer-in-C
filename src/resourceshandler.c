#include "resourceshandler.h"

Resources _Resources(Resources *parent,char * name)
{
    return (Resources){
        .path = parent?makePath(parent->path,name):strdup(name),
        .getResource = getResource,
        .toString = toString,
        .return_data = 0
    };
}

Resources getResource(Resources *parent,char * resName,int isFile)
{
    Resources res = 
    {
        .getResource = getResource,
        .toString = toString,
        .return_data = 0
    };
    if (!parent)
    {
        res.path = strdup(resName);
    }
    else{
        res.path =  makePath(parent->path,resName);
    }
    if(!res.path) return (Resources){0};

    if(isFile)
        res.return_data = mmapFile(res.path);
    return res;
}

char *toString(Resources *parent){
    if(!parent) return strdup("res");
    return parent->path;
}

char *makePath_(int count, char *paths[]) {
    size_t alloc_size = BUFF_SIZE;
    char *path = malloc(alloc_size);
    if (!path) {
        MemoryError();
        return NULL;
    }

    size_t size = 0;
    for (int i = 0; i < count; i++) {
        size_t s = strlen(paths[i]);
        if (size + s + 1 > alloc_size) {
            alloc_size = size + s + 1 + 16; 
            char *tmp = realloc(path, alloc_size);
            if (!tmp) {
                free(path);
                MemoryError();
                return NULL;
            }
            path = tmp;
        }

        memcpy(path + size, paths[i], s);
        size += s;

        if (i < count - 1) {
            path[size++] = '/';
        }
    }

    path[size] = '\0';

    char *tmp = realloc(path, size + 1);
    if (tmp) path = tmp; 

    return path;
}



void destroyResources(Resources * res){
    free(res->path);
    if(res->return_data){
        #ifdef _WIN32
            UnmapViewOfFile(res->return_data);
        #else
            size_t s = strlen(res->return_data);
            munmap(res->return_data,s);
        #endif
    }
}