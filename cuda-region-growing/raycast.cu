#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "bmp.h"

// data is 3D, total size is DATA_DIM x DATA_DIM x DATA_DIM
#define DATA_DIM 512
#define DATA_SIZE (DATA_DIM * DATA_DIM * DATA_DIM) 
#define DATA_SIZE_BYTES (sizeof(unsigned char) * DATA_SIZE)

// image is 2D, total size is IMAGE_DIM x IMAGE_DIM
#define IMAGE_DIM 512
#define IMAGE_SIZE (IMAGE_DIM * IMAGE_DIM)
#define IMAGE_SIZE_BYTES (sizeof(unsigned char) * IMAGE_SIZE)

texture<char, cudaTextureType3D, cudaReadModeNormalizedFloat> data_texture;
texture<char, cudaTextureType3D, cudaReadModeElementType> region_texture;

void print_time(struct timeval start, struct timeval end){
    long int ms = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    double s = ms/1e6;
    printf("Time : %f s\n", s);
}
// Stack for the serial region growing
typedef struct{
    int size;
    int buffer_size;
    int3* pixels;
} stack_t;

stack_t* new_stack(){
    stack_t* stack = (stack_t*)malloc(sizeof(stack_t));
    stack->size = 0;
    stack->buffer_size = 1024;
    stack->pixels = (int3*)malloc(sizeof(int3)*1024);

    return stack;
}

void push(stack_t* stack, int3 p){
    if(stack->size == stack->buffer_size){
        stack->buffer_size *= 2;
        int3* temp = stack->pixels;
        stack->pixels = (int3*)malloc(sizeof(int3)*stack->buffer_size);
        memcpy(stack->pixels, temp, sizeof(int3)*stack->buffer_size/2);
        free(temp);

    }
    stack->pixels[stack->size] = p;
    stack->size += 1;
}

int3 pop(stack_t* stack){
    stack->size -= 1;
    return stack->pixels[stack->size];
}

// float3 utilities
__host__ __device__ float3 cross(float3 a, float3 b){
    float3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

__host__ __device__ float3 normalize(float3 v){
    float l = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    v.x /= l;
    v.y /= l;
    v.z /= l;

    return v;
}

__host__ __device__ float3 add(float3 a, float3 b){
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;

    return a;
}

__host__ __device__ float3 scale(float3 a, float b){
    a.x *= b;
    a.y *= b;
    a.z *= b;

    return a;
}


// Prints CUDA device properties
void print_properties(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);

    cudaDeviceProp p;
    cudaSetDevice(0);
    cudaGetDeviceProperties (&p, 0);
    printf("Compute capability: %d.%d\n", p.major, p.minor);
    printf("Name: %s\n" , p.name);
    printf("\n\n");
}


// Fills data with values
unsigned char func(int x, int y, int z){
    unsigned char value = rand() % 20;

    int x1 = 300;
    int y1 = 400;
    int z1 = 100;
    float dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));

    if(dist < 100){
        value  = 30;
    }

    x1 = 100;
    y1 = 200;
    z1 = 400;
    dist = sqrt((x-x1)*(x-x1) + (y-y1)*(y-y1) + (z-z1)*(z-z1));



    if(dist < 50){
        value = 50;
    }

    if(x > 200 && x < 300 && y > 300 && y < 500 && z > 200 && z < 300){
        value = 45;
    }
    if(x > 0 && x < 100 && y > 250 && y < 400 && z > 250 && z < 400){
        value =35;
    }
    return value;
}

unsigned char* create_data(){
    unsigned char* data = (unsigned char*)malloc(sizeof(unsigned char) * DATA_DIM*DATA_DIM*DATA_DIM);

    for(int i = 0; i < DATA_DIM; i++){
        for(int j = 0; j < DATA_DIM; j++){
            for(int k = 0; k < DATA_DIM; k++){
                data[i*DATA_DIM*DATA_DIM + j*DATA_DIM+ k]= func(k,j,i);
            }
        }
    }

    return data;
}

// Checks if position is inside the volume (float3 and int3 versions)
__host__ __device__ int inside(float3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM-1);
    int y = (pos.y >= 0 && pos.y < DATA_DIM-1);
    int z = (pos.z >= 0 && pos.z < DATA_DIM-1);

    return x && y && z;
}

__host__ __device__ int inside(int3 pos){
    int x = (pos.x >= 0 && pos.x < DATA_DIM);
    int y = (pos.y >= 0 && pos.y < DATA_DIM);
    int z = (pos.z >= 0 && pos.z < DATA_DIM);

    return x && y && z;
}

// Indexing function (note the argument order)
__host__ __device__ int index(int z, int y, int x){
    return z * DATA_DIM*DATA_DIM + y*DATA_DIM + x;
}

// Trilinear interpolation
__host__ __device__ float value_at(float3 pos, unsigned char* data){
    if(!inside(pos)){
        return 0;
    }

    int x = floor(pos.x);
    int y = floor(pos.y);
    int z = floor(pos.z);

    int x_u = ceil(pos.x);
    int y_u = ceil(pos.y);
    int z_u = ceil(pos.z);

    float rx = pos.x - x;
    float ry = pos.y - y;
    float rz = pos.z - z;

    float a0 = rx*data[index(z,y,x)] + (1-rx)*data[index(z,y,x_u)];
    float a1 = rx*data[index(z,y_u,x)] + (1-rx)*data[index(z,y_u,x_u)];
    float a2 = rx*data[index(z_u,y,x)] + (1-rx)*data[index(z_u,y,x_u)];
    float a3 = rx*data[index(z_u,y_u,x)] + (1-rx)*data[index(z_u,y_u,x_u)];

    float b0 = ry*a0 + (1-ry)*a1;
    float b1 = ry*a2 + (1-ry)*a3;

    float c0 = rz*b0 + (1-rz)*b1;


    return c0;
}


// Serial ray casting
unsigned char* raycast_serial(unsigned char* data, unsigned char* region){
    unsigned char* image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);

    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    // For each pixel
    for(int y = -(IMAGE_DIM/2); y < (IMAGE_DIM/2); y++){
        for(int x = -(IMAGE_DIM/2); x < (IMAGE_DIM/2); x++){

            // Find the ray for this pixel
            float3 screen_center = add(camera, forward);
            float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
            ray = add(ray, scale(camera, -1));
            ray = normalize(ray);
            float3 pos = camera;

            // Move along the ray, we stop if the color becomes completely white,
            // or we've done 5000 iterations (5000 is a bit arbitrary, it needs 
            // to be big enough to let rays pass through the entire volume)
            int i = 0;
            float color = 0;
            while(color < 255 && i < 5000){
                i++;
                pos = add(pos, scale(ray, step_size));          // Update position
                int r = value_at(pos, region);                  // Check if we're in the region
                color += value_at(pos, data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
            }

            // Write final color to image
            image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
        }
    }

    return image;
}


// Check if two values are similar, threshold can be changed.
__host__ __device__ int similar(unsigned char* data, int3 a, int3 b){
    unsigned char va = data[a.z * DATA_DIM*DATA_DIM + a.y*DATA_DIM + a.x];
    unsigned char vb = data[b.z * DATA_DIM*DATA_DIM + b.y*DATA_DIM + b.x];

    int i = abs(va-vb) < 1;
    return i;
}


// Serial region growing, same algorithm as in assignment 2
unsigned char* grow_region_serial(unsigned char* data){
    unsigned char* region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM*DATA_DIM*DATA_DIM);

    stack_t* stack = new_stack();

    int3 seed = {.x=50, .y=300, .z=300};
    push(stack, seed);
    region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 1;

    int dx[6] = {-1,1,0,0,0,0};
    int dy[6] = {0,0,-1,1,0,0};
    int dz[6] = {0,0,0,0,-1,1};

    while(stack->size > 0){
        int3 pixel = pop(stack);
        for(int n = 0; n < 6; n++){
            int3 candidate = pixel;
            candidate.x += dx[n];
            candidate.y += dy[n];
            candidate.z += dz[n];

            if(!inside(candidate)){
                continue;
            }

            if(region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
                continue;
            }

            if(similar(data, pixel, candidate)){
                push(stack, candidate);
                region[candidate.z * DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 1;
            }
        }
    }

    return region;
}


__global__ void raycast_kernel(unsigned char* data, unsigned char* image, unsigned char* region){
    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    int x = blockIdx.x - (IMAGE_DIM/2);
    int y = threadIdx.x - (IMAGE_DIM/2);

    // Find the ray for this pixel
    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;

    // Move along the ray
    int i = 0;
    float color = 0;
    while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));          // Update position
        int r = value_at(pos, region);                  // Check if we're in the region
        color += value_at(pos, data)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
    }

    // Write final color to image
    image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;
}


__global__ void raycast_kernel_texture(unsigned char* image){
    // Camera/eye position, and direction of viewing. These can be changed to look
    // at the volume from different angles.
    float3 camera = {.x=1000,.y=1000,.z=1000};
    float3 forward = {.x=-1, .y=-1, .z=-1};
    float3 z_axis = {.x=0, .y=0, .z = 1};

    // Finding vectors aligned with the axis of the image
    float3 right = cross(forward, z_axis);
    float3 up = cross(right, forward);

    // Creating unity lenght vectors
    forward = normalize(forward);
    right = normalize(right);
    up = normalize(up);

    float fov = 3.14/4;
    float pixel_width = tan(fov/2.0)/(IMAGE_DIM/2);
    float step_size = 0.5;

    int x = blockIdx.x - (IMAGE_DIM/2);
    int y = threadIdx.x - (IMAGE_DIM/2);

    // Find the ray for this pixel
    float3 screen_center = add(camera, forward);
    float3 ray = add(add(screen_center, scale(right, x*pixel_width)), scale(up, y*pixel_width));
    ray = add(ray, scale(camera, -1));
    ray = normalize(ray);
    float3 pos = camera;

    // Move along the ray
    int i = 0;
    float color = 0;
    while(color < 255 && i < 5000){
        i++;
        pos = add(pos, scale(ray, step_size));          // Update position
        int r = tex3D(region_texture, pos.x, pos.y, pos.z);    // Look up value from texture
        if(inside(pos)){
            color += 255 * tex3D(data_texture, pos.x, pos.y, pos.z)*(0.01 + r) ;       // Update the color based on data value, and if we're in the region
        }
    }

    // Write final color to image
    image[(y+(IMAGE_DIM/2)) * IMAGE_DIM + (x+(IMAGE_DIM/2))] = color > 255 ? 255 : color;

}


unsigned char* raycast_gpu(unsigned char* data, unsigned char* region){
    //Declare and allocate device memory
    unsigned char* device_image;
    unsigned char* device_data;
    unsigned char* device_region;

    cudaMalloc(&device_image, IMAGE_SIZE_BYTES); 
    cudaMalloc(&device_data, DATA_SIZE_BYTES);
    cudaMalloc(&device_region, DATA_SIZE_BYTES);

    //Copy data to the device
    cudaMemcpy(device_data, data, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(device_region, region, DATA_SIZE_BYTES, cudaMemcpyHostToDevice);

    //Run the kernel
    raycast_kernel<<<IMAGE_DIM, IMAGE_DIM>>>(device_data, device_image, device_region);  

    //Allocate memory for the result
    unsigned char* host_image = (unsigned char*)malloc(IMAGE_SIZE_BYTES);

    //Copy result from device
    cudaMemcpy(host_image, device_image, IMAGE_SIZE_BYTES, cudaMemcpyDeviceToHost);
    
    //Free device memory
    cudaFree(device_region);
    cudaFree(device_data);
    cudaFree(device_image);
    return host_image;
}


unsigned char* raycast_gpu_texture(unsigned char* data, unsigned char* region){
    data_texture.filterMode = cudaFilterModeLinear; //Interpolate the data texture
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
    cudaExtent extent = make_cudaExtent(DATA_DIM, DATA_DIM, DATA_DIM);

    //Allocate arrays
    cudaArray* data_array;
    cudaArray* region_array;
    cudaMalloc3DArray(&region_array, &channelDesc, extent, 0);
    cudaMalloc3DArray(&data_array, &channelDesc, extent, 0);
    
    //Copy data to region array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(region, sizeof(char) * IMAGE_DIM, IMAGE_DIM, IMAGE_DIM);
    copyParams.dstArray = region_array;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    //Copy data to data array
    copyParams.srcPtr   = make_cudaPitchedPtr(data, sizeof(char) * IMAGE_DIM, IMAGE_DIM, IMAGE_DIM);
    copyParams.dstArray = data_array;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    //Bind arrays to the textures
    cudaBindTextureToArray(data_texture, data_array);
    cudaBindTextureToArray(region_texture, region_array);

    //Allocate memory for the result on the device
    unsigned char* device_image;
    cudaMalloc(&device_image, IMAGE_SIZE_BYTES); 

    raycast_kernel_texture<<<IMAGE_DIM, IMAGE_DIM>>>(device_image);  

    //Allocate memory to retrieve the result
    unsigned char* host_image = (unsigned char*)malloc(sizeof(unsigned char)*IMAGE_DIM*IMAGE_DIM);

    //Fetch the result
    cudaMemcpy(host_image, device_image, IMAGE_SIZE_BYTES, cudaMemcpyDeviceToHost);

    //Unbind textures
    cudaUnbindTexture(data_texture);
    cudaUnbindTexture(region_texture);
    
    //Free memory on the device
    cudaFreeArray(data_array);
    cudaFreeArray(region_array);
    cudaFree(device_image);

    return host_image;
}


__global__ void region_grow_kernel(unsigned char* data, unsigned char* region, int* unfinished){
    int3 pixel = {.x = blockIdx.x * blockDim.x + threadIdx.x
                 ,.y = blockIdx.y * blockDim.y + threadIdx.y
                 ,.z = blockIdx.z * blockDim.z + threadIdx.z
                };

    int index = (pixel.z * DATA_DIM * DATA_DIM) + (pixel.y * DATA_DIM) + pixel.x;

    if(region[index] == 2){
        //Race conditions should not matter, as we only write 1s, and if one of them gets through it's enough
        *unfinished = 1; 
        region[index] = 1;

        int dx[6] = {-1,1,0,0,0,0};
        int dy[6] = {0,0,-1,1,0,0};
        int dz[6] = {0,0,0,0,-1,1};
        


        for(int n = 0; n < 6; n++){
            int3 candidate;
            candidate.x = pixel.x + dx[n];
            candidate.y = pixel.y + dy[n];
            candidate.z = pixel.z + dz[n];

            if(!inside(candidate)){
                continue;
            }

            if(region[candidate.z*DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x]){
                continue;
            }

            if(similar(data, pixel, candidate)){
                region[candidate.z*DATA_DIM*DATA_DIM + candidate.y*DATA_DIM + candidate.x] = 2;
            }
        }

    }

}

__device__ bool is_border(int3 pixel, int dim){
    if( pixel.x == 0 || pixel.y == 0 || pixel.z == 0){
        return true;
    }
    if( pixel.x == dim - 1 || pixel.y == dim - 1 || pixel.z == dim - 1){
        return true;
    }

    return false;
}

__global__ void region_grow_kernel_shared(unsigned char* data, unsigned char* region, int* unfinished){
    int local_region_dim = blockDim.x; //We use the knowledge that in this problem dim_x = dim_y = dim_z

    extern __shared__ char local_region[];
    __shared__ bool block_done;


    int3 local_pixel = {.x = threadIdx.x  
                       ,.y = threadIdx.y
                       ,.z = threadIdx.z
    };

    int local_index = (local_pixel.z * local_region_dim * local_region_dim) + (local_pixel.y * local_region_dim) + local_pixel.x;

    int3 global_pixel = {.x = blockIdx.x * blockDim.x + threadIdx.x - 1
            ,.y = blockIdx.y * blockDim.y + threadIdx.y - 1
            ,.z = blockIdx.z * blockDim.z + threadIdx.z - 1
    };

    int global_index = (global_pixel.z * DATA_DIM * DATA_DIM) + (global_pixel.y * DATA_DIM) + global_pixel.x;

    if(global_index >= DATA_DIM * DATA_DIM * DATA_DIM){
        return;
    }
    if(global_index < 0){
        return;
    }

    local_region[local_index] = region[global_index];
    do{
        block_done = true;
        __syncthreads();

        if(local_region[local_index] == 2 && !is_border(local_pixel, local_region_dim)){
            local_region[local_index] = 1;

            int dx[6] = {-1,1,0,0,0,0};
            int dy[6] = {0,0,-1,1,0,0};
            int dz[6] = {0,0,0,0,-1,1};

            for(int n = 0; n < 6; n++){
                int3 candidate;
                candidate.x = local_pixel.x + dx[n];
                candidate.y = local_pixel.y + dy[n];
                candidate.z = local_pixel.z + dz[n];

                int3 global_candidate;
                global_candidate.x = global_pixel.x + dx[n];
                global_candidate.y = global_pixel.y + dy[n];
                global_candidate.z = global_pixel.z + dz[n];

                int candidate_local_index = (candidate.z * local_region_dim * local_region_dim) 
                                            + (candidate.y * local_region_dim)
                                            + candidate.x;

                if(local_region[candidate_local_index] != 0){
                    continue;
                }

                //if(similar(data, global_pixel, global_candidate)){
                    local_region[candidate_local_index] = 2;
                    block_done = false;
                    *unfinished = 1;
               // }
            }
        }
        __syncthreads();
    }while(!block_done);

    if(is_border(local_pixel, local_region_dim)){
        if(local_region[local_index] == 2 && local_pixel.y == 0){ //Only copy the 2s from the border
            region[global_index] = 2;
        }
    }else{
        region[global_index] = local_region[local_index];
    }
}

unsigned char* grow_region_gpu(unsigned char* host_data){
    int region_size = sizeof(unsigned char) * DATA_DIM*DATA_DIM*DATA_DIM;
    int data_size = region_size;

    unsigned char* host_region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM*DATA_DIM*DATA_DIM);
    
    int*            host_unfinished = (int*) calloc(sizeof(int), 1);

    unsigned char*  device_region;
    unsigned char*  device_data;
    int*            device_unfinished;

    cudaMalloc(&device_region, region_size);
    cudaMalloc(&device_data, data_size);
    cudaMalloc(&device_unfinished, 1);

    //plant seed
    int3 seed = {.x=50, .y=300, .z=300};
    host_region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 2;

    cudaMemcpy(device_region, host_region, region_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data, host_data, data_size, cudaMemcpyHostToDevice);


    dim3 block_size;
    block_size.x = 7;
    block_size.y = 7;
    block_size.z = 7;

    dim3 grid_size;
    grid_size.x = DATA_DIM / block_size.x + 1; // Add 1 to round up instead of down.
    grid_size.y = DATA_DIM / block_size.y + 1;
    grid_size.z = DATA_DIM / block_size.z + 1;
    int i = 0;

    printf("Getting ready to start that loop \n");

    do{
        i++;
        *host_unfinished = 0;
        cudaMemcpy(device_unfinished, host_unfinished, 1, cudaMemcpyHostToDevice);
        region_grow_kernel<<<grid_size, block_size>>>(device_data, device_region, device_unfinished);
        cudaMemcpy(host_unfinished, device_unfinished, 1, cudaMemcpyDeviceToHost);
    }while(*host_unfinished);

    printf("Ran %d iterations\n",i);

    cudaMemcpy(host_region, device_region, region_size, cudaMemcpyDeviceToHost);

    cudaFree(device_region);
    cudaFree(device_data);
    cudaFree(device_unfinished);

    return host_region;
}


unsigned char* grow_region_gpu_shared(unsigned char* host_data){
    int region_size = sizeof(unsigned char) * DATA_DIM * DATA_DIM * DATA_DIM;
    int data_size = region_size;

    unsigned char* host_region = (unsigned char*)calloc(sizeof(unsigned char), DATA_DIM * DATA_DIM * DATA_DIM);
    int*            host_unfinished = (int*) calloc(sizeof(int), 1);

    unsigned char*  device_region;
    unsigned char*  device_data;
    int*            device_unfinished;

    cudaMalloc(&device_region, region_size);
    cudaMalloc(&device_data, data_size);
    cudaMalloc(&device_unfinished, sizeof(int));

    //plant seed
    int3 seed = {.x=50, .y=300, .z=300};
    host_region[seed.z *DATA_DIM*DATA_DIM + seed.y*DATA_DIM + seed.x] = 2;

    cudaMemcpy(device_region, host_region, region_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_data, host_data, data_size, cudaMemcpyHostToDevice);


    dim3 block_size;
    block_size.x = 9;
    block_size.y = 9;
    block_size.z = 9;

    dim3 grid_size;
    grid_size.x = DATA_DIM / (block_size.x - 2) + 1; // - 2 to include the borders 
    grid_size.y = DATA_DIM / (block_size.y - 2) + 1;
    grid_size.z = DATA_DIM / (block_size.z - 2) + 1;
    
    int local_region_dim = block_size.x; //We use the knowledge that in this problem dim_x = dim_y = dim_z
    int local_region_size = local_region_dim * local_region_dim * local_region_dim;

    int i = 0;

    printf("Getting ready to start that loop \n");

    do{
        printf("loop\n");
        i++;
        *host_unfinished = 0;
        cudaMemcpy(device_unfinished, host_unfinished, 1, cudaMemcpyHostToDevice);
        region_grow_kernel_shared<<<grid_size, block_size, sizeof(char) * local_region_size>>>(device_data, device_region, device_unfinished);
        cudaMemcpy(host_unfinished, device_unfinished, 1, cudaMemcpyDeviceToHost);
    //}while(*host_unfinished != 0);
    }while(i<20);

    printf("Ran %d iterations\n",i);

    cudaMemcpy(host_region, device_region, region_size, cudaMemcpyDeviceToHost);

    cudaFree(device_region);
    cudaFree(device_data);
    cudaFree(device_unfinished);

    return host_region;
}


int main(int argc, char** argv){
    struct timeval start, end;

    print_properties();


    gettimeofday(&start, NULL);
    unsigned char* data = create_data();
    printf("Create data time:\n");
    gettimeofday(&end, NULL);
    print_time(start, end);

    gettimeofday(&start, NULL);
    unsigned char* region = grow_region_gpu_shared(data);
    gettimeofday(&end, NULL);
    printf("Grow time:\n");
    print_time(start, end);
    printf("Errors: %s\n", cudaGetErrorString(cudaGetLastError()));

    gettimeofday(&start, NULL);
    unsigned char* image = raycast_gpu_texture(data, region);
    gettimeofday(&end, NULL);
    printf("Raycast time: \n");
    print_time(start, end);
    printf("Errors: %s\n", cudaGetErrorString(cudaGetLastError()));

    free(data);
    free(region);

    gettimeofday(&start, NULL);
    write_bmp(image, IMAGE_DIM, IMAGE_DIM);
    gettimeofday(&end, NULL);
    printf("bmp time: \n");
    print_time(start, end);
}
