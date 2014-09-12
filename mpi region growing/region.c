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

const int TAG = 2;

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
             haloless_col_t,
             halo_col_t,
             region_t;


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

    int a = im[p.x + p.y * (local_image_size[1] + 2)];
    int b = im[q.x + q.y * (local_image_size[1] + 2)];
    int diff = abs(a-b);
    return diff < 2;

}

//TODO: Remove
int malloc2dchar(char ***array, int n, int m) {

    /* allocate the n*m contiguous items */
    char *p = (char *)malloc(n*m*sizeof(char));
    if (!p) return -1;

    /* allocate the row pointers into the memory */
    (*array) = (char **)malloc(n*sizeof(char*));
    if (!(*array)) {
        free(p);
        return -1;
    }

    /* set up the pointers into the contiguous memory */
    for (int i=0; i<n; i++)
        (*array)[i] = &(p[i*m]);

    return 0;
}

// Create and commit MPI datatypes
void create_types(){
    int starts[2]   = {0,0}; 

    MPI_Type_vector(local_image_size[1],
                    1,
                    image_size[0],
                    MPI_UNSIGNED_CHAR,
                    &haloless_col_t);

    MPI_Type_vector(local_image_size[1],
                    1,
                    local_image_size[0] + 2,
                    MPI_UNSIGNED_CHAR,
                    &halo_col_t);
    
    MPI_Type_commit(&haloless_col_t);
    MPI_Type_commit(&halo_col_t);

}


// Send image from rank 0 to all ranks, from image to local_image
void distribute_image(){
    
    if(rank == 0){
        for(int x = 0; x < dims[0]; x++){
            for(int y = 0; y < dims[1]; y++){
                int send_rank = 0;
                int send_coords[2] = {y,x};
                MPI_Cart_rank(cart_comm, send_coords, &send_rank);
                printf("Sending to (%d, %d),  rank %d.\n", x, y, send_rank);

                //Send all the rows excluding halo
                for(int i = 0; i < local_image_size[1]; i++){
                    char* start 
                        = &image[x*local_image_size[0]
                                +y*local_image_size[1]*image_size[0]
                                +i*image_size[0]];

                    MPI_Send(start, local_image_size[0]
                            ,MPI_UNSIGNED_CHAR, send_rank
                            ,TAG, cart_comm);
                }
                //Send halo columns
            }
        }
    }

    MPI_Status status;
    //Receive all the rows excluding halo
    for(int i = 0; i < local_image_size[1]; i++){
        char* start = 
            &local_image[
            (i+1)*(local_image_size[0] + 2)
            +1];

        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, 0,
                TAG, cart_comm, &status);
    }

}

void distribute_image_halo(){
    if(rank == 0){
        for(int x = 0; x < dims[0]; x++){
            for(int y = 0; y < dims[1]; y++){
                int send_rank;
                int send_coords[2] = {y,x};
                MPI_Cart_rank(cart_comm, send_coords, &send_rank);

                //Send north halo 
                if(y > 0){
                    char* start 
                        = &image[
                        x*local_image_size[0]              //x grid colums
                        +(y-1)*local_image_size[1] * image_size[0]  //y-1 grid rows
                        +(local_image_size[0]-1)*image_size[0]];    //Fetch last row

                    MPI_Send(start, local_image_size[0]
                            ,MPI_UNSIGNED_CHAR, send_rank
                            ,TAG, cart_comm);

                }
                
                //Send south halo 
                if(y < dims[1] - 1){
                    char* start 
                        = &image[
                        x*local_image_size[0]                           //x grid colums
                        +(y+1)*local_image_size[1] * image_size[0]];   //y+1 grid rows

                    MPI_Send(start, local_image_size[0]
                            ,MPI_UNSIGNED_CHAR, send_rank
                            ,TAG, cart_comm);

                }
                
                //Send west halo 
                if(x > 0){
                    char* col_start 
                        = &image[(x - 1) * local_image_size[0]  //x - 1 grid colums
                        + y * local_image_size[1]  * image_size[0]  //y grid rows
                        + (local_image_size[0]-1)];    //Fetch last col

                    MPI_Send(col_start, 1
                            ,haloless_col_t, send_rank
                            ,TAG, cart_comm);

                }
                
                //Send east halo 
                if(x < dims[0] - 1){
                    char* col_start 
                        = &image[(x + 1) * local_image_size[0]  //x + 1 grid colums
                        + y * local_image_size[1]  * image_size[0]];  //y grid rows

                    MPI_Send(col_start, 1
                            ,haloless_col_t, send_rank
                            ,TAG, cart_comm);
                }
            }
        }
    }

    MPI_Status status;
    if(north >= 0){
        char* start = 
            &local_image[1];

        printf("Receiving north (%d,%d) %d \n ", coords[0],coords[1], rank);
        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, 0,
                TAG, cart_comm, &status);
        printf("got north %d \n ", rank);
    }

    if(south >= 0){
        char* start = 
            &local_image[
                       (local_image_size[0]+1) * (local_image_size[1]+2) //Last row
                        + 1]; //Skip the corner

        printf("Receiving south (%d,%d) %d \n ", coords[0],coords[1], rank);
        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, 0,
                TAG, cart_comm, &status);
        printf("got south %d \n ", rank);
    }

    if(west >= 0){
        char* col_start = 
            &local_image[local_image_size[0]+2];

        printf("Receiving west (%d,%d) %d \n ", coords[0],coords[1], rank);
        MPI_Recv(col_start,
                1, halo_col_t, 0,
                TAG, cart_comm, &status);
        printf("got west %d \n ", rank);
    }

    if(east >= 0){
        char* col_start = 
            &local_image[
                (local_image_size[0] + 2) * 2 - 1];

        printf("Receiving west (%d,%d) %d \n ", coords[0],coords[1], rank);
        MPI_Recv(col_start,
                1, halo_col_t, 0,
                TAG, cart_comm, &status);
        printf("got west %d \n ", rank);
    }
}


// Exchange borders with neighbour ranks
void exchange(stack_t* stack){
   //Odd rows/cols send first, then even.

    MPI_Status status;
    if(coords[1] % 2 == 0){
        char* start = 
            &local_region[local_image_size[0]+3];
        MPI_Send(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, north,
                TAG, cart_comm);

        start = 
            &local_region[(local_image_size[1] + 1) * (local_image_size[0] + 2) + 1];
        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, south,
                TAG, cart_comm, &status);

        start = 
            &local_region[(local_image_size[1] + 0) * (local_image_size[0] + 2) + 1];
        MPI_Send(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, south,
                TAG, cart_comm);

        start = 
            &local_region[1];
        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, north,
                TAG, cart_comm, &status);
    }else{
        char* start = 
            &local_region[(local_image_size[1] + 1) * (local_image_size[0] + 2) + 1];
        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, south,
                TAG, cart_comm, &status);

        start = 
            &local_region[local_image_size[0]+3];
        MPI_Send(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, north,
                TAG, cart_comm);

        start = 
            &local_region[1];
        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, north,
                TAG, cart_comm, &status);

        start = 
            &local_region[(local_image_size[1] + 0) * (local_image_size[0] + 2) + 1];
        MPI_Send(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, south,
                TAG, cart_comm);
    }
    
    if(coords[1] % 2 == 0){
        char* start = 
            &local_region[local_image_size[0]+3];
        MPI_Send(start,
                1, halo_col_t, west,
                TAG, cart_comm);

        start = 
            &local_region[(local_image_size[0] + 2) * 2 - 1];
        MPI_Recv(start,
                1, halo_col_t, east,
                TAG, cart_comm, &status);

        start = 
            &local_region[(local_image_size[0] + 2) * 2 - 2];
        MPI_Send(start,
                1, halo_col_t, east,
                TAG, cart_comm);

        start = 
            &local_region[local_image_size[0] + 2];
        MPI_Recv(start,
                1, halo_col_t, west,
                TAG, cart_comm, &status);
    }else{
        char* start = 
            &local_region[(local_image_size[0] + 2) * 2 - 1];
        MPI_Recv(start,
                1, halo_col_t, east,
                TAG, cart_comm, &status);

        start = 
            &local_region[local_image_size[0]+3];
        MPI_Send(start,
                1, halo_col_t, west,
                TAG, cart_comm);

        start = 
            &local_region[local_image_size[0] + 2];
        MPI_Recv(start,
                1, halo_col_t, west,
                TAG, cart_comm, &status);

        start = 
            &local_region[(local_image_size[0] + 2) * 2 - 2];
        MPI_Send(start,
                1, halo_col_t, east,
                TAG, cart_comm);
    }
}


// Gather region bitmap from all ranks to rank 0, from local_region to region
void gather_region(){
    //All ranks send to 0 in order using barriers
    for(int i = 0; i < size; i++){
        if(rank == i){
            for(int y = 0; y < local_image_size[1]; y++){
                char* start = 
                    &local_region[
                    (y + 1) * (local_image_size[0] + 2)// send row nr y (+1 for halo)
                    + 1 ]; // Avoid halo

                MPI_Send(start, local_image_size[0],
                        MPI_UNSIGNED_CHAR, 0, TAG, cart_comm);

            }
        }
        MPI_Barrier(cart_comm);
    }

    if(rank == 0){
        for(int recv_rank = 0; recv_rank < size; recv_rank++){
            int recv_coords[2] = {0,0};
            MPI_Cart_coords(cart_comm, recv_rank, size, &recv_rank);
            int x = recv_coords[0];
            int y = recv_coords[1];
            printf(" yolo %d, %d \n", x,y);

            MPI_Status status;

            char* start =
                &region[y * image_size[0] + x * local_image_size[0]];

            printf ("Invalid? %d\n", start);
            MPI_Recv(start, local_image_size[0],
                    MPI_UNSIGNED_CHAR, recv_rank, TAG, cart_comm, &status);
        }
    }


}

// Determine if all ranks are finished. You may have to add arguments.
// You dont have to have this check as a seperate function
int finished(){
   
}


// Check if pixel is inside local image
int inside(pixel_t p){
    return (p.x > 0 && p.x <= local_image_size[1] && p.y > 0 && p.y <= local_image_size[0]);
}


// Adding seeds in corners.
void add_seeds(stack_t* stack){
    int seed_pos = 2;
    int seeds [8];

    //Reordered for easier editing
    seeds[0] = seed_pos;
    seeds[1] = seed_pos;
    seeds[3] = seed_pos;
    seeds[6] = seed_pos;
    seeds[2] = local_image_size[1]+1-seed_pos;
    seeds[4] = local_image_size[1]+1-seed_pos;
    seeds[5] = local_image_size[0]+1-seed_pos;
    seeds[7] = local_image_size[0]+1-seed_pos;
    
    for(int i = 0; i < 4; i++){
        pixel_t seed;
        seed.x = seeds[i*2];
        seed.y = seeds[i*2+1];
        
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
        
        local_region[pixel.y * (local_image_size[1] + 2) + pixel.x] = 1;
        
        
        int dx[4] = {0,0,1,-1}, dy[4] = {1,-1,0,0};
        for(int c = 0; c < 4; c++){
            pixel_t candidate;
            candidate.x = pixel.x + dx[c];
            candidate.y = pixel.y + dy[c];
            
            if(!inside(candidate)){
                continue;
            }
            
            
            if(local_region[candidate.y * (local_image_size[1] + 2) + candidate.x]){
                continue;
            }
            
            if(similar(local_image, pixel, candidate)){
                local_region[candidate.y * (local_image_size[1] + 2) + candidate.x] = 1;
                push(stack,candidate);
            }
        }
    }
    exchange(stack);
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

//TODO: delete
void generate_debug_image(){
    image_size[0] = 8;
    image_size[1] = 8;

    local_image_size[0] = image_size[0]/dims[0];
    local_image_size[1] = image_size[1]/dims[1];

    int lsize = local_image_size[0]*local_image_size[1];
    int lsize_border = (local_image_size[0] + 2)*(local_image_size[1] + 2);
    local_image = (unsigned char*)calloc(sizeof(unsigned char),lsize_border);
    local_region = (unsigned char*)calloc(sizeof(unsigned char),lsize_border);


    srand(1337);
    if(rank == 0){
        printf("what is happening? \n");
        image = (unsigned char*)calloc(sizeof(unsigned char) , image_size[0]*image_size[1]);
        for (int i=0; i<image_size[1]; i++) {
            for (int j=0; j<image_size[0]; j++)
                //image[j +  i * image_size[0]] = rand()%8;
                image[j +  i * image_size[0]] = j;
        }    

        printf("Image! \n", rank);
        for (int i=0; i<image_size[0]; i++) {
            putchar('|');
            for (int j=0; j<image_size[1]; j++) {
                //putchar(image[j +  i * image_size[0]]);
                printf("%d ", image[j +  i * image_size[0]]);
            }
            printf("|\n");
        }

        printf("Image layout! \n", rank);
        for (int j=0; j<image_size[0]*image_size[1]; j++) {
            //putchar(image[j]);
            printf("%d ",image[j]);
        }
        printf("\n");
    }
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

void print_debug_info(){
    MPI_Barrier(MPI_COMM_WORLD);
    for (int p=0; p<size; p++) {
        if (rank == p) {
            printf("Local process on rank %d is:\n", rank);
            for (int i=0; i<local_image_size[0]+2; i++) {
                putchar('|');
                for (int j=0; j<local_image_size[1]+2; j++) {
                    printf("%d ",local_image[j +  i * (local_image_size[0]+2)]);
                }
                printf("|\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void print_regions(){

    for (int p=0; p<size; p++) {
        if (rank == p) {
            printf("Region on rank %d is:\n", rank);
            for (int i=0; i<local_image_size[0]+2; i++) {
                putchar('|');
                for (int j=0; j<local_image_size[1]+2; j++) {
                    printf("%d ",local_region[j +  i * (local_image_size[0]+2)]);
                }
                printf("|\n");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


int main(int argc, char** argv){
    
    init_mpi(argc, argv);
    
    generate_debug_image();
    load_and_allocate_images(argc, argv);
    
    create_types();
    
    distribute_image();

    distribute_image_halo();

    grow_region();
    
    //print_debug_info();
    //print_regions();
    gather_region();
    
    MPI_Finalize();
    
    //write_image();
    
    exit(0);
}
