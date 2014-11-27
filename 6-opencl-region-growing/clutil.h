#ifndef CLUTIL_H
#define CLUTIL_H
#include <CL/cl.h>

const char *clErrorStr(cl_int err);
void clError(char *s, cl_int err);
void printPlatformInfo(cl_platform_id platform);
void printDeviceInfo(cl_device_id device);
cl_kernel buildKernel(char* sourceFile, char* kernelName, char* options, cl_context context, cl_device_id device);

#endif
