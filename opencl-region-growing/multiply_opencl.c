#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "clutil.h"

#define SIZE 1024



void host_multiply(float* a, float* b, float* result) {
    for(int i = 0; i < 1024; i++){
        result[i] = a[i] * b[i];
    }
}

int main() {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_int err;
    char *source;
    int i;
    
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    
    printPlatformInfo(platform);
    printDeviceInfo(device);
    
    queue = clCreateCommandQueue(context, device, 0, &err);
    kernel = buildKernel("multiply_opencl.cl", "multiply", NULL, context, device);
    
    float* a_host = (float*)malloc(sizeof(float)*1024);
    float* b_host = (float*)malloc(sizeof(float)*1024);
    
    for(int i = 0; i < 1024; i++){
        a_host[i] = i+1;
        b_host[i] = 1.0/(float)(i+1);
    }
    
    float* result_host = (float*)malloc(sizeof(float)*1024);
    
    host_multiply(a_host, b_host, result_host);
    
    cl_mem a_device = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE*sizeof(cl_float),NULL,&err);
    cl_mem b_device = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE*sizeof(cl_float),NULL,&err);
    cl_mem result_device = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                          SIZE*sizeof(cl_float),NULL,&err);
    clError("Error allocating memory", err);
    
    clEnqueueWriteBuffer(queue, b_device, CL_FALSE, 0, SIZE*sizeof(cl_float), b_host, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, a_device, CL_FALSE, 0, SIZE*sizeof(cl_float), a_host, 0, NULL, 
                         NULL);
    
    err = clSetKernelArg(kernel, 0, sizeof(a_device), (void*)&a_device);
    err = clSetKernelArg(kernel, 1, sizeof(b_device), (void*)&b_device);
    err = clSetKernelArg(kernel, 2, sizeof(result_device), (void*)&result_device);
    clError("Error setting arguments", err);
    
    size_t globalws=SIZE;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalws, NULL, 0, NULL, NULL);
    
    clFinish(queue);
    float* result_from_device = (float*)malloc(sizeof(float)*1024);
    err = clEnqueueReadBuffer(queue, result_device, CL_TRUE, 0, SIZE*sizeof(cl_float), 
                              result_from_device, 0, NULL, NULL);
    clFinish(queue);
    
    printf("Host\tDevice\n");
    for(int i = 0; i < 10; i++){
        printf("%0.2f\t%0.2f\n" , result_host[i], result_from_device[i]);
    }
    
    
    clReleaseMemObject(a_device);
    clReleaseMemObject(b_device);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
