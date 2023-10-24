#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void my_Bcast(double *data, int size, MPI_Datatype datatype, MPI_Comm comm, int N){
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    MPI_Request request;
    MPI_Status status;
    int max_timestep = ceil(log2(num_procs)) - 1;
    int t;

    for (t = 0; t <= max_timestep; t++) {
        int tot_nodes = pow(2, t);
        int rank_start = pow(2, t);
        int src_rank, dst_rank;

        for (src_rank = 0; src_rank < tot_nodes; src_rank++) {
            dst_rank = src_rank + tot_nodes;
            if (rank == src_rank && dst_rank < num_procs) {
                MPI_Isend(data, size, datatype, dst_rank, 0, comm, &request);
            }
        }

        for (dst_rank = rank_start; dst_rank < (rank_start + tot_nodes); dst_rank++) {
            src_rank = dst_rank - tot_nodes;
            if (rank == dst_rank && src_rank < num_procs) {
                MPI_Irecv(data, size, datatype, src_rank, 0, comm, &request);
                MPI_Wait(&request, &status);
            }
        }
    }
// Check the sum
    double sum = 0;
    for (int i = 0; i < N; i++){
        sum += data[i];
    }
    printf("my rank %d, sum = %f\n", rank, sum);
//    int *int_data = (int *)data; // Cast the void pointer to int pointer
//    for (int i = 0; i < size; i++) {
//        sum += int_data[i];
//    }
//    printf("my rank %d, sum = %f\n", rank, sum);
}

int main (int argc, char *argv[]) {
    int rank_main, size;
    double time_start, time_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_main);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
// Test data size
    int N = pow(2, 14); //datapoints
    double *data = (double *)malloc(N * sizeof(double));
 
    if (rank_main == 0) {
 // Initialize data on the root process
        for (int i = 0; i < N; i++) {
            data[i] = (double)i;
        }
     }
  // Time detection
     MPI_Barrier(MPI_COMM_WORLD);
     time_start = MPI_Wtime();
 // Perform custom broadcast
     my_Bcast(data, N, MPI_DOUBLE, MPI_COMM_WORLD, N);

 // Now, data on all processes is updated
// Time detection
    MPI_Barrier(MPI_COMM_WORLD);
    time_end = MPI_Wtime();

    if (rank_main==0) {
        printf("time: %f\n", time_end - time_start);
    }
 // Test MPI_Bcast
     double *data_mpi = (double *)malloc(N * sizeof(double));
     if (rank_main == 0) {
         for (int i = 0; i < N; i++) {
             data_mpi[i] = (double)i;
         }
    }
    MPI_Bcast(data_mpi, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       
    MPI_Finalize();
//    printf("success!");
}
