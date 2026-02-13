#ifndef FILE_HANDLER
#define FILE_HANDLER

#include "datatypes.h"

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
#endif
void *mmapFile(const char *path);
long count_line(char *data);
#endif