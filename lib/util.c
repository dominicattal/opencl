#include "util.h"
#include <stdio.h>
#include <stdlib.h>

char* read_file(const char* path)
{
    FILE* file = NULL;
    char* buf = NULL;
    long pos;
    int ret;

    file = fopen(path, "r");
    if (file == NULL)
        goto fail;

    ret = fseek(file, 0, SEEK_END);
    if (ret != 0)
        goto fail;

    pos = ftell(file);
    if (pos == -1L)
        goto fail;

    ret = fseek(file, 0, SEEK_SET);
    if (ret != 0)
        goto fail;

    buf = malloc((pos+1) * sizeof(char));
    if (buf == NULL)
        goto fail;

    ret = fread(buf, sizeof(char), pos, file);
    if (ferror(file) != 0)
        goto fail;

    buf[ret] = '\0';

    return buf;

fail:
    if (file != NULL)
        fclose(file);
    if (buf != NULL)
        free(buf);
    return NULL;
}
