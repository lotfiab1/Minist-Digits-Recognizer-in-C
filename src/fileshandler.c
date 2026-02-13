#include "fileshandler.h"

void *mmapFile(const char *path)
{
    void *data = NULL; 
    #ifdef _WIN32
        HANDLE hFile = CreateFileA(
            path, GENERIC_READ, FILE_SHARE_READ,
            NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

        if (hFile == INVALID_HANDLE_VALUE)
            return NULL;


        HANDLE hMap = CreateFileMappingA(
            hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        CloseHandle(hFile);
        if (!hMap) return NULL;
        
        data = MapViewOfFile(
            hMap, FILE_MAP_READ, 0, 0, 0);
            
        
        CloseHandle(hMap);
    #else
        int fd = open(path,O_RDONLY);
        if(fd < 0){
            printf("Failed open file\n");
            return NULL;
        }
        struct stat sb;
        fstat(fd, &sb);
        data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (data == MAP_FAILED) {
            printf("Failed Map file %s\n",path);
            return NULL;
        }
        close(fd);
    #endif

    return data;
}

long count_line(char *data)
{
    long count = 0;

    for (const char *p = data; *p; p++) {
        if (*p == '\n')
            count++;
    }

    if (count > 0 && data[0] != '\0' && data[strlen(data) - 1] != '\n')
        count++;

    return count;
}