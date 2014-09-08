#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include "bmp.h"

typedef struct{
    int x;
    int y;
} pixel_t;

typedef struct{
    int size;
    int buffer_size;
    pixel_t* pixels;
} stack_t;


// Global variables
int rank,                       // MPI rank
    size,                       // Number of MPI processes
    dims[2],                    // Dimensions of MPI grid
    coords[2],                  // Coordinate of this rank in MPI grid
    periods[2] = {0,0},         // Periodicity of grid
    north,south,east,west,      // Four neighbouring MPI ranks
    image_size[2] = {512,512},  // Hard coded image size
    local_image_size[2];        // Size of local part of image (not including border)


MPI_Comm cart_comm;             // Cartesian communicator


// MPI datatypes, you may have to add more.
MPI_Datatype border_row_t,
             border_col_t;


unsigned char *image,           // Entire image, only on rank 0
              *region,          // Region bitmap. 1 if in region, 0 elsewise
              *local_image,     // Local part of image
              *local_region;    // Local part of region bitmap


// Create new pixel stack
stack_t* new_stack(){
    stack_t* stack = (stack_t*)malloc(sizeof(stack_t));
    stack->size = 0;
    stack->buffer_size = 1024;
    stack->pixels = (pixel_t*)malloc(sizeof(pixel_t*)*1024);
}


// Push on pixel stack
void push(stack_t* stack, pixel_t p){
    if(stack->size == stack->buffer_size){
        stack->buffer_size *= 2;
        stack->pixels = realloc(stack->pixels, sizeof(pixel_t)*stack->buffer_size);
    }
    stack->pixels[stack->size] = p;
    stack->size += 1;
}


// Pop from pixel stack
pixel_t pop(stack_t* stack){
    stack->size -= 1;
    return stack->pixels[stack->size];
}


// Check if two pixels are similar. The hardcoded threshold can be changed.
// More advanced similarity checks could have been used.
int similar(unsigned char* im, pixel_t p, pixel_t q){
    int a = im[p.x +  p.y * image_size[1]];
    int b = im[q.x +  q.y * image_size[1]];
    int diff = abs(a-b);
    return diff < 2;
}


// Create and commit MPI datatypes
void create_types(){
    
}


// Send image from rank 0 to all ranks, from image to local_image
void distribute_image(){
    
    
}


// Exchange borders with neighbour ranks
void exchange(stack_t* stack){
    
    
}


// Gather region bitmap from all ranks to rank 0, from local_region to region
void gather_region(){
    
    
}

// Determine if all ranks are finished. You may have to add arguments.
// You dont have to have this check as a seperate function
int finished(){
   
}


// Check if pixel is inside local image
int inside(pixel_t p){
    return (p.x >= 0 && p.x < local_image_size[1] && p.y >= 0 && p.y < local_image_size[0]);
}


// Adding seeds in corners.
void add_seeds(stack_t* stack){
    int seeds [8];
    seeds[0] = 5;
    seeds[1] = 5;
    seeds[2] = image_size[1]-5;
    seeds[3] = 5;
    seeds[4] = image_size[1]-5;
    seeds[5] = image_size[0]-5;
    seeds[6] = 5;
    seeds[7] = image_size[0]-5;
    
    for(int i = 0; i < 4; i++){
        pixel_t seed;
        seed.x = seeds[i*2] - coords[1]*local_image_size[1];
        seed.y = seeds[i*2+1] -coords[0]*local_image_size[0];
        
        if(inside(seed)){
            push(stack, seed);
        }
    }
}


// Region growing, serial implementation
void grow_region(){
    
    stack_t* stack = new_stack();
    add_seeds(stack);
        
    while(stack->size > 0){
        pixel_t pixel = pop(stack);
        
        region[pixel.y * image_size[1] + pixel.x] = 1;
        
        
        int dx[4] = {0,0,1,-1}, dy[4] = {1,-1,0,0};
        for(int c = 0; c < 4; c++){
            pixel_t candidate;
            candidate.x = pixel.x + dx[c];
            candidate.y = pixel.y + dy[c];
            
            if(!inside(candidate)){
                continue;
            }
            
            
            if(region[candidate.y * image_size[1] + candidate.x]){
                continue;
            }
            
            if(similar(image, pixel, candidate)){
                region[candidate.x + candidate.y * image_size[1]] = 1;
                push(stack,candidate);
            }
        }
    }
}


// MPI initialization, setting up cartesian communicator
void init_mpi(int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm );
    MPI_Cart_coords( cart_comm, rank, 2, coords );
    
    MPI_Cart_shift( cart_comm, 0, 1, &north, &south );
    MPI_Cart_shift( cart_comm, 1, 1, &west, &east );
}


void load_and_allocate_images(int argc, char** argv){

    if(argc != 2){
        printf("Useage: region file");
        exit(-1);
    }
    
    if(rank == 0){
        image = read_bmp(argv[1]);
        region = (unsigned char*)calloc(sizeof(unsigned char),image_size[0]*image_size[1]);
    }
    
    local_image_size[0] = image_size[0]/dims[0];
    local_image_size[1] = image_size[1]/dims[1];
    
    int lsize = local_image_size[0]*local_image_size[1];
    int lsize_border = (local_image_size[0] + 2)*(local_image_size[1] + 2);
    local_image = (unsigned char*)malloc(sizeof(unsigned char)*lsize_border);
    local_region = (unsigned char*)calloc(sizeof(unsigned char),lsize_border);
}


void write_image(){
    if(rank==0){
        for(int i = 0; i < image_size[0]*image_size[1]; i++){

            image[i] *= (region[i] == 0);
        }
        write_bmp(image, image_size[0], image_size[1]);
    }
}


int main(int argc, char** argv){
    
    init_mpi(argc, argv);
    
    load_and_allocate_images(argc, argv);
    
    create_types();
    
    distribute_image();

    grow_region();
    
    gather_region();
    
    MPI_Finalize();
    
    write_image();
    
    exit(0);
}
