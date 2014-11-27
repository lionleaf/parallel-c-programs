#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "bmp.h"

const int image_width = 512;
const int image_height = 512;
const int image_size = 512*512;
const int color_depth = 255;

volatile unsigned char* image;
volatile unsigned char* output_image;
int n_threads;

volatile int* histogram;
volatile float* transfer_function;

int counter;
pthread_mutex_t mutex;
pthread_cond_t cond_var;


void barrier(){
    pthread_mutex_lock(&mutex);
    counter++;
    if(counter == n_threads){
        counter = 0;
        pthread_cond_broadcast(&cond_var);
    } else{
        while(pthread_cond_wait(&cond_var, &mutex)!=0);
    }
    pthread_mutex_unlock(&mutex);
}

void* run_thread(void* thread_id){
    long t_id = (long) thread_id;

    
    int* local_histogram = (int*)calloc(sizeof(int), color_depth);
    for(int i = t_id; i < image_size; i += n_threads){
        local_histogram[image[i]]++;
    }


    pthread_mutex_lock(&mutex);
    for(int i = 0; i < color_depth; i++){
        histogram[i] += local_histogram[i];
    }
    pthread_mutex_unlock(&mutex);

    barrier();

    for(int i = t_id; i < color_depth; i += n_threads){
        for(int j = 0; j < i+1; j++){
            transfer_function[i] += color_depth * ((float)histogram[j])/(image_size);
        }
    }
    
    barrier();

    for(int i = t_id; i < image_size; i += n_threads){
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
    histogram = (int*)calloc(sizeof(int), color_depth);
    transfer_function = (float*)calloc(sizeof(float), color_depth);

    pthread_t threads[n_threads];

    for(long i = 0; i < n_threads; i++){
        pthread_create(&threads[i], NULL, run_thread, (void*) i);
    }
    for(int i = 0; i < n_threads; i++){
        pthread_join(threads[i], NULL);
    }
    write_bmp(output_image, image_width, image_height);
    pthread_exit(NULL);
}
