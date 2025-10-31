#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

void print_platform_info_string(cl_platform_id id, cl_platform_info param_name)
{
    cl_int err;
    char* value_str;
    size_t value_str_size;

    err = clGetPlatformInfo(id, param_name, 0, NULL, &value_str_size);
    if (err != CL_SUCCESS) {
        puts("Could not get value string size");
        exit(1);
    }

    value_str = malloc(value_str_size);
    err = clGetPlatformInfo(id, param_name, value_str_size, value_str, NULL);
    if (err != CL_SUCCESS) {
        puts("Could not get value string");
        exit(1);
    }

    puts(value_str);
    free(value_str);
}

void print_platform_info(cl_platform_id id)
{
    static cl_platform_info param_names[] = {
        CL_PLATFORM_PROFILE,
        CL_PLATFORM_NAME,
        CL_PLATFORM_VERSION,
        CL_PLATFORM_VENDOR,
        CL_PLATFORM_EXTENSIONS
    };
    int n = sizeof(param_names) / sizeof(cl_platform_info);
    for (int i = 0; i < n; i++)
        print_platform_info_string(id, param_names[i]);
    puts("");
}

int main()
{
    cl_int err;
    cl_uint num_platforms;
    cl_uint num_entries;
    cl_platform_id* platforms;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS) {
        puts("Could not find number of platforms");
        exit(1);
    }

    num_entries = num_platforms;
    platforms = malloc(num_entries * sizeof(cl_platform_id));
    clGetPlatformIDs(num_entries, platforms, NULL);
    if (err != CL_SUCCESS) {
        puts("Could not find platforms");
        exit(1);
    }

    for (cl_uint i = 0; i < num_entries; i++)
        print_platform_info(platforms[i]);

    free(platforms);

    return 0;
}
