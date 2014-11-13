#include "clutil.h"
#include <CL/cl.h>
#include <stdio.h>

const char *clErrorStr(cl_int err) {
	switch (err) {
	case CL_SUCCESS:                          return "Success!";
	case CL_DEVICE_NOT_FOUND:                 return "Device not found.";
	case CL_DEVICE_NOT_AVAILABLE:             return "Device not available";
	case CL_COMPILER_NOT_AVAILABLE:           return "Compiler not available";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return "Memory object allocation failure";
	case CL_OUT_OF_RESOURCES:                 return "Out of resources";
	case CL_OUT_OF_HOST_MEMORY:               return "Out of host memory";
	case CL_PROFILING_INFO_NOT_AVAILABLE:     return "Profiling information not available";
	case CL_MEM_COPY_OVERLAP:                 return "Memory copy overlap";
	case CL_IMAGE_FORMAT_MISMATCH:            return "Image format mismatch";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return "Image format not supported";
	case CL_BUILD_PROGRAM_FAILURE:            return "Program build failure";
	case CL_MAP_FAILURE:                      return "Map failure";
	case CL_INVALID_VALUE:                    return "Invalid value";
	case CL_INVALID_DEVICE_TYPE:              return "Invalid device type";
	case CL_INVALID_PLATFORM:                 return "Invalid platform";
	case CL_INVALID_DEVICE:                   return "Invalid device";
	case CL_INVALID_CONTEXT:                  return "Invalid context";
	case CL_INVALID_QUEUE_PROPERTIES:         return "Invalid queue properties";
	case CL_INVALID_COMMAND_QUEUE:            return "Invalid command queue";
	case CL_INVALID_HOST_PTR:                 return "Invalid host pointer";
	case CL_INVALID_MEM_OBJECT:               return "Invalid memory object";
	case CL_INVALID_IMAGE_SIZE:               return "Invalid image size";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return "Invalid image format descriptor";
	case CL_INVALID_SAMPLER:                  return "Invalid sampler";
	case CL_INVALID_BINARY:                   return "Invalid binary";
	case CL_INVALID_BUILD_OPTIONS:            return "Invalid build options";
	case CL_INVALID_PROGRAM:                  return "Invalid program";
	case CL_INVALID_PROGRAM_EXECUTABLE:       return "Invalid program executable";
	case CL_INVALID_KERNEL_NAME:              return "Invalid kernel name";
	case CL_INVALID_KERNEL_DEFINITION:        return "Invalid kernel definition";
	case CL_INVALID_KERNEL:                   return "Invalid kernel";
	case CL_INVALID_ARG_INDEX:                return "Invalid argument index";
	case CL_INVALID_ARG_VALUE:                return "Invalid argument value";
	case CL_INVALID_ARG_SIZE:                 return "Invalid argument size";
	case CL_INVALID_KERNEL_ARGS:              return "Invalid kernel arguments";
	case CL_INVALID_WORK_DIMENSION:           return "Invalid work dimension";
	case CL_INVALID_WORK_GROUP_SIZE:          return "Invalid work group size";
	case CL_INVALID_WORK_ITEM_SIZE:           return "Invalid work item size";
	case CL_INVALID_GLOBAL_OFFSET:            return "Invalid global offset";
	case CL_INVALID_EVENT_WAIT_LIST:          return "Invalid event wait list";
	case CL_INVALID_EVENT:                    return "Invalid event";
	case CL_INVALID_OPERATION:                return "Invalid operation";
	case CL_INVALID_GL_OBJECT:                return "Invalid OpenGL object";
	case CL_INVALID_BUFFER_SIZE:              return "Invalid buffer size";
	case CL_INVALID_MIP_LEVEL:                return "Invalid mip-map level";
	default:                                  return "Unknown";
	}
}
void clError(char *s, cl_int err) {
    if(err != CL_SUCCESS){
	    fprintf(stderr,"%s: %s\n",s,clErrorStr(err));
	    exit(1);
    }
}

void printPlatformInfo(cl_platform_id platform){
    cl_int err;
    char paramValue[100];
    size_t returnSize;

    printf("===== Platform info ====\n");

    err = clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, 100, paramValue, &returnSize);
    printf("Platform profile:\t%s\n", paramValue);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 100, paramValue, &returnSize);
    printf("Platform version:\t%s\n", paramValue);

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 100, paramValue, &returnSize);
    printf("Platform name:\t\t%s\n", paramValue);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 100, paramValue, &returnSize);
    printf("Platform vendor:\t%s\n", paramValue);

    err = clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, 100, paramValue, &returnSize);
    printf("Platform extensions:\t");
    if(returnSize > 0){
        printf("%s\n", paramValue);
    }
    else{
        printf("(none)");
    }

    printf("\n\n");
}

void printDeviceInfo(cl_device_id device){
    cl_int err;
    char paramValueString[100];
    cl_ulong paramValueUlong;
    cl_uint paramValueUint;
    size_t returnSize;

    printf("==== Device info ====\n");

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 100, paramValueString, &returnSize);
    printf("Device name:\t\t%s\n", paramValueString);

    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &paramValueUlong, &returnSize);
    printf("Device global memory:\t%lu\n", paramValueUlong);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &paramValueUint, &returnSize);
    printf("Device max frequency:\t%u\n", paramValueUint);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &paramValueUint, &returnSize);
    printf("Device compute units:\t%u\n", paramValueUint);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &paramValueUlong, &returnSize);
    printf("Max size of memory allocation:\t%lu\n", paramValueUlong);
    
    err = clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &paramValueUint, &returnSize);
    printf("Device memory alignment:\t%u\n", paramValueUint);

    printf("\n\n");
}

char *load_program_source(const char *s) {
	char *t;
	size_t len;
	FILE *f = fopen(s, "r");
	if(NULL== f){
        fprintf(stderr,"couldn't open file");
        exit(0);
    }
	fseek(f,0,SEEK_END);
	len=ftell(f);
	fseek(f,0,SEEK_SET);
	t=malloc(len+1);
	size_t bb = fread(t,len,1,f);
	t[len]=0;
	fclose(f);
	return t;
}

cl_kernel buildKernel(char* sourceFile, char* kernelName, char* options, cl_context context, cl_device_id device){
    cl_int err;
    
    char* source = load_program_source(sourceFile);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
    clError("Error creating program",err);
    
    err = clBuildProgram(program, 1, &device, options, NULL, NULL);
    if(CL_SUCCESS != err) {
        static char s[1048576];
        size_t len;
        //clError("Error building program", err);
        fprintf(stderr,"Error building program\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(s), s, &len);
        fprintf(stderr,"Build log:\n%s\n", s);
        exit(1);
    }
    
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    clError("Error creating kernel",err);
    free(source);

    return kernel;
}
