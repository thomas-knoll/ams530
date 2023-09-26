#include <stdio.h>
#include <mpi.h>
int main (int argc, char *argv[]) {
int rank, size, number;
MPI_Init (&argc, &argv);
MPI_Comm_rank (MPI_COMM_WORLD, &rank);
MPI_Comm_size (MPI_COMM_WORLD, &size);
MPI_Request request;
if (rank == 0){
printf( "Hello from %d of %d\n", rank, size );
MPI_Isend(&rank, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &request);
//this following line indicated hat the completion of the requests should be ignored
request = MPI_REQUEST_NULL;
}
else{
int number = 0;
MPI_Irecv(&number, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &request);
printf( "Hello from %d of %d\n", rank, size );
if (rank < size-1)
MPI_Send(&rank, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);}

MPI_Finalize();
}
