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

    int a = im[p.x + p.y * (local_image_size[0] + 2)];
    int b = im[q.x + q.y * (local_image_size[0] + 2)];
    int diff = abs(a-b);
    return diff < 2;

}

// Create and commit MPI datatypes
void create_types(){
    MPI_Type_vector(local_image_size[0],
                    1,
                    image_size[1],
                    MPI_UNSIGNED_CHAR,
                    &haloless_col_t);

    MPI_Type_vector(local_image_size[0],
                    1,
                    local_image_size[1] + 2,
                    MPI_UNSIGNED_CHAR,
                    &halo_col_t);
    
    MPI_Type_commit(&haloless_col_t);
    MPI_Type_commit(&halo_col_t);

}


// Send image from rank 0 to all ranks, from image to local_image
void distribute_image(){
    
    if(rank == 0){
        for(int x = 0; x < dims[1]; x++){
            for(int y = 0; y < dims[0]; y++){
                int send_rank = 0;
                int send_coords[2] = {y,x};
                MPI_Cart_rank(cart_comm, send_coords, &send_rank);

                //Send all the rows excluding halo
                for(int i = 0; i < local_image_size[1]; i++){
                    char* start 
                        = &image[x*local_image_size[0]
                                +y*local_image_size[0]*image_size[0]
                                +i*image_size[0]];

                    MPI_Send(start, local_image_size[0]
                            ,MPI_UNSIGNED_CHAR, send_rank
                            ,TAG, cart_comm);
                }
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
        for(int x = 0; x < dims[1]; x++){
            for(int y = 0; y < dims[0]; y++){
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
                if(y < dims[0] - 1){
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
                if(x < dims[1] - 1){
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

        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, 0,
                TAG, cart_comm, &status);
    }

    if(south >= 0){
        char* start = 
            &local_image[
                       (local_image_size[0]+1) * (local_image_size[1]+2) //Last row
                        + 1]; //Skip the corner

        MPI_Recv(start,
                local_image_size[0], MPI_UNSIGNED_CHAR, 0,
                TAG, cart_comm, &status);
    }

    if(west >= 0){
        char* col_start = 
            &local_image[local_image_size[0]+2];

        MPI_Recv(col_start,
                1, halo_col_t, 0,
                TAG, cart_comm, &status);
    }

    if(east >= 0){
        char* col_start = 
            &local_image[
                (local_image_size[0] + 2) * 2 - 1];

        MPI_Recv(col_start,
                1, halo_col_t, 0,
                TAG, cart_comm, &status);
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

void add_halo_to_stack(stack_t* stack){
    for(int rows = 0; rows < 2; rows++){
        for(int x = 0; x < local_image_size[0] + 2; x++){
            pixel_t pixel;
            pixel.x = x;
            pixel.y = rows?
                local_image_size[1]+1
                :0;
            if(local_region[x + pixel.y * (local_image_size[0] + 2)]){
                push(stack, pixel);
            }
        }
    }
    for(int y = 0; y < local_image_size[1] + 2; y++){
        pixel_t pixel;
        pixel.x = local_image_size[0] + 1;
        pixel.y = y; 

        if(local_region[pixel.x + pixel.y * (local_image_size[0] + 2)]){
            push(stack, pixel);
        }
    }

    for(int y = 0; y < local_image_size[1] + 2; y++){
        pixel_t pixel;
        pixel.x = 0;
        pixel.y = y; 

        if(local_region[pixel.x + pixel.y * (local_image_size[0] + 2)]){
            push(stack, pixel);
        }
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
    
    // Receive the data
    if(rank == 0){
        for(int recv_rank = 0; recv_rank < size; recv_rank++){
            int recv_coords[2] = {0,0};
            MPI_Cart_coords(cart_comm, recv_rank, size, recv_coords);
            int x = recv_coords[1];
            int y = recv_coords[0];

            MPI_Status status;

            for(int row = 0; row < local_image_size[1]; row++){
                char* start =
                    &region[y * local_image_size[0] * image_size[0] //move down to block
                            + row * image_size[0]               //move rows down
                            + x * local_image_size[0]];         //Move x blocks right

                MPI_Recv(start, local_image_size[0],
                        MPI_UNSIGNED_CHAR, recv_rank, TAG, cart_comm, &status);
            }
        }
    }


}

// Determine if all ranks are finished. You may have to add arguments.
int finished(int local_finished){
    int global_finished;
    MPI_Allreduce(&local_finished, &global_finished, 
            1, MPI_INT, MPI_MIN, cart_comm);
    return global_finished;
}


// Check if pixel is inside local image
int inside(pixel_t p){
    return (p.x > 0 && p.x <= local_image_size[0] && p.y > 0 && p.y <= local_image_size[1]);
}


// Adding seeds in corners.
void add_seeds(stack_t* stack){
    int seed_pos = 5;
    int seeds [8];

    //Reordered for easier editing
    seeds[0] = seed_pos;
    seeds[1] = seed_pos;
    seeds[2] = local_image_size[1]-seed_pos;
    seeds[3] = seed_pos;
    seeds[4] = local_image_size[1]-seed_pos;
    seeds[5] = local_image_size[0]-seed_pos;
    seeds[6] = seed_pos;
    seeds[7] = local_image_size[0]-seed_pos;

    int ranks[4] = {0,0,0,0};
    ranks[0] = 0;
    ranks[2] = size - 1;
 
    int lower_right;
    int xy[2] = {0,dims[1] - 1};
    MPI_Cart_rank(cart_comm, xy, &lower_right);
    ranks[1] = lower_right;

    int upper_left; //upper left in the bmp 
    xy[0] = dims[0] - 1;
    xy[1] = 0;
    MPI_Cart_rank(cart_comm, xy, &upper_left);
    ranks[3] = upper_left;
    
    
    for(int i = 0; i < 4; i++){
        if(rank != ranks[i]) continue;
        pixel_t seed;
        seed.x = seeds[i*2];
        seed.y = seeds[i*2+1];
        
        if(inside(seed)){
            push(stack, seed);
        }
    }
}


void grow_region(){
    stack_t* stack = new_stack();
    add_seeds(stack);
    int local_finish = 0;
    while(!finished(local_finish)){
        local_finish = 1;
        while(stack->size > 0){
            pixel_t pixel = pop(stack);


            local_region[pixel.y * (local_image_size[0] + 2) + pixel.x] = 1;


            int dx[4] = {0,0,1,-1}, dy[4] = {1,-1,0,0};
            for(int c = 0; c < 4; c++){
                pixel_t candidate;
                candidate.x = pixel.x + dx[c];
                candidate.y = pixel.y + dy[c];

                if(!inside(candidate)){
                    continue;
                }


                if(local_region[candidate.y * (local_image_size[0] + 2) + candidate.x]){
                    continue;
                }

                if(similar(local_image, pixel, candidate)){
                    local_region[candidate.y * (local_image_size[0] + 2) + candidate.x] = 1;
                    push(stack,candidate);
                    local_finish = 0;
                }
            }
        }
        exchange(stack);
        add_halo_to_stack(stack);

        MPI_Barrier(cart_comm);
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

    distribute_image_halo();

    grow_region();
    
    gather_region();
    
    MPI_Finalize();
    
    write_image();
    
    exit(0);
}
