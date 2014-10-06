#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "bmp.h"

const int image_width = 512;
const int image_height = 512;
const int image_size = 512*512;
const int color_depth = 255;

unsigned char* image;
unsigned char* output_image;
int n_threads;


void* run_thread(void* thread_id){
    long t_id = (long) thread_id;

    printf("Running on thread %ld of %d \n", (long) thread_id, n_threads);
    
    int chunk_size = image_size / n_threads ;

    int* histogram = (int*)calloc(sizeof(int), color_depth);
    for(int i = chunk_size * t_id; i < chunk_size * (t_id + 1); i++){
        histogram[image[i]]++;
    }

    float* transfer_function = (float*)calloc(sizeof(float), color_depth);
    for(int i = 0; i < color_depth; i++){
        for(int j = 0; j < i+1; j++){
            transfer_function[i] += color_depth*((float)histogram[j])/(image_size);
        }
    }


    for(int i = 0; i < image_size; i++){
        output_image[i] = transfer_function[image[i]];
    }

    pthread_exit(NULL);

}

int main(int argc, char** argv){


    if(argc != 3){
        printf("Useage: %s image n_threads\n", argv[0]);
        exit(-1);
    }
    n_threads = atoi(argv[2]);

    image = read_bmp(argv[1]);
    output_image = malloc(sizeof(unsigned char) * image_size);
    
    pthread_t threads[n_threads];

    for(long i = 0; i < n_threads; i++){
        pthread_create(&threads[i], NULL, run_thread, (void*) i);
    }

    write_bmp(output_image, image_width, image_height);
    pthread_exit(NULL);
}
