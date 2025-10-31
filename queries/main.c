#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

void print_platform_device_info_type(cl_device_id device, const char* fmt)
{
    cl_int err;
    cl_device_type value;

    err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &value, NULL);
    if (err != CL_SUCCESS) {
        puts("Could not get device info type");
        exit(1);
    }

    const char* s;
    switch (value) {
        case CL_DEVICE_TYPE_CPU:
            s = "CPU";
            break;
        case CL_DEVICE_TYPE_GPU:
            s = "GPU";
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            s = "ACCELERATOR";
            break;
        case CL_DEVICE_TYPE_CUSTOM:
            s = "CUSTOM";
            break;
        default:
            s = "UNKNOWN";
            break;
    }

    printf(fmt, s);
}

void print_platform_device_info_uint(cl_device_id device, cl_device_info param_name, const char* fmt)
{
    cl_int err;
    cl_uint value;

    err = clGetDeviceInfo(device, param_name, sizeof(cl_uint), &value, NULL);
    if (err != CL_SUCCESS) {
        puts("Could not get device info type");
        exit(1);
    }

    printf(fmt, value); 
}

void print_platform_device_info(cl_device_id device)
{
    static const char* fmt[] = {
        "Type:                      %s\n",
        "Vendor ID:                 %04x\n",
        "Max Compute Units:         %d\n",
        "Max Work Item Dimensions:  %d\n",
        "Max Clock Frequency:       %d\n",
    };
    print_platform_device_info_type(device, fmt[0]);
    print_platform_device_info_uint(device, CL_DEVICE_VENDOR_ID, fmt[1]);
    print_platform_device_info_uint(device, CL_DEVICE_MAX_COMPUTE_UNITS, fmt[2]);
    print_platform_device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, fmt[3]);
    print_platform_device_info_uint(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, fmt[4]);
    puts("");
}

void platform_device_info(cl_platform_id platform)
{
    cl_int err;
    cl_uint num_devices;
    cl_device_id* devices;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS) {
        puts("Could not get the number of platform devices");
        exit(1);
    }
    
    devices = malloc(num_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        puts("Could not get platform devices");
        exit(1);
    }

    for (int i = 0; i < (int)num_devices; i++)
        print_platform_device_info(devices[i]);

    free(devices);
}

void print_platform_info_string(cl_platform_id platform, cl_platform_info param_name, const char* fmt)
{
    cl_int err;
    char* value_str;
    size_t value_str_size;

    err = clGetPlatformInfo(platform, param_name, 0, NULL, &value_str_size);
    if (err != CL_SUCCESS) {
        puts("Could not get value string size");
        exit(1);
    }

    value_str = malloc(value_str_size);
    err = clGetPlatformInfo(platform, param_name, value_str_size, value_str, NULL);
    if (err != CL_SUCCESS) {
        puts("Could not get value string");
        exit(1);
    }

    printf(fmt, value_str);
    free(value_str);
}

void print_platform_info(cl_platform_id platform)
{
    int n;
    static const char* fmt[] = {
        "Profile:    %s\n",
        "Name:       %s\n",
        "Version:    %s\n", 
        "Vendor:     %s\n",
        "Extensions: %s\n"
    };
    static cl_platform_info param_names[] = {
        CL_PLATFORM_PROFILE,
        CL_PLATFORM_NAME,
        CL_PLATFORM_VERSION,
        CL_PLATFORM_VENDOR,
        CL_PLATFORM_EXTENSIONS
    };

    n = sizeof(param_names) / sizeof(cl_platform_info);
    for (int i = 0; i < n; i++)
        print_platform_info_string(platform, param_names[i], fmt[i]);
    puts("");
    
    platform_device_info(platform);

    puts("==================================");
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
