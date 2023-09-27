#include <stdio.h>
#include <mpi.h>
int main (int argc, char *argv[]) {
    int rank, size;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Status status;
    if (rank == 0){
        int rank_next = rank + 1;
        printf("Hello from Processor %d of %d\n", rank, size);
        MPI_Send(&rank, 1, MPI_INT, rank_next, 0, MPI_COMM_WORLD);}
    else{
        int rank_prev = rank - 1;
        int rank_next = rank + 1;
        int message = 0;
        MPI_Recv(&message, 1, MPI_INT, rank_prev, 0, MPI_COMM_WORLD, &status);
        printf("Hello from Processor %d of %d\n", rank, size);
        if (rank < size-1){
            MPI_Send(&rank, 1, MPI_INT, rank_next, 0, MPI_COMM_WORLD);
        }
    }
    MPI_Finalize();
}
