#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  int size, rank;
  int msg = 0;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  
  
  if(rank != 0){
    //Wait for message from below
    MPI_Recv(&msg, 1, MPI_INT, rank - 1, 1, MPI_COMM_WORLD, &status);      
    printf("Rank %d received %d \n", rank, msg);
    //Increment
    msg+=1;
  }
  
  if(rank < size - 1){
    //Send message up
    MPI_Send(&msg, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD);      
    printf("Rank %d sent %d \n", rank, msg);

    //Wait for message to come down again
    MPI_Recv(&msg, 1, MPI_INT, rank+1, 1, MPI_COMM_WORLD, &status);      
    printf("Rank %d received %d \n", rank, msg);
    
    //Increment again
    msg += 1;
  }
  
  if(rank != 0){
    //Send down again
    MPI_Send(&msg, 1, MPI_INT, rank-1, 1, MPI_COMM_WORLD);      
    printf("Rank %d sent %d \n", rank, msg);
  }


  MPI_Finalize();
}
