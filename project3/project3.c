#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char *argv[] )
{
    int rank, size;
    double** M;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    printf("Hello from %d of %d processors.\n", rank, size);
    MPI_Finalize();
    //initialize a matrix of size 1000x1000 with dynamic memory allocation and random dooulbe precision floating point numbers between -1 and 1
    M = (double **)malloc(1000 * sizeof(double *));
    for (int i=0; i<1000; i++)
        M[i] = (double *)malloc(1000 * sizeof(double));
    for (int i=0; i<1000; i++)
        for (int j=0; j<1000; j++)
            M[i][j] = (double)rand()/RAND_MAX*2.0-1.0;
    //print the matrix into a file called matrix.txt
    FILE *fp;   //file pointer
    fp = fopen("matrix.txt", "w");
    for (int i=0; i<1000; i++)
    {
        for (int j=0; j<1000; j++)
            fprintf(fp, "%f ", M[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    free(M);
}
