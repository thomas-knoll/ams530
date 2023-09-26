#include <stdio.h>
#include <mpi.h>
int main (int argc, char *argv[]) {
int rank, size, number;
MPI_Init (&argc, &argv);
MPI_Comm_rank (MPI_COMM_WORLD, &rank);
MPI_Comm_size (MPI_COMM_WORLD, &size);
if (rank == 0){
printf( "Hello from %d of %d\n", rank, size );
MPI_Send(&rank, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);}
else{
int number = 0;
MPI_Recv(&number, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
printf( "Hello from %d of %d\n", rank, size );
if (rank < size-1)
MPI_Send(&rank, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);}

MPI_Finalize();
}
