#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

void seq_mat_mult(double* mat1, double* mat2, double* mat3, int n)
{
    for (int i=0; i<n; i++)
        for (int j=0; j<n; j++)
            for (int k=0; k<n; k++)
                mat3[i*n+j] += mat1[i*n+k] * mat2[k*n+j];
}

void parallel_mat_mult(double* mat1, double* mat2, double* mat3, int n, int size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calculate local matrix size
     int local_n = n / size;

    // Allocate memory for local matrices
    double* local_A = (double*)malloc(local_n * n * sizeof(double));
    double* local_B = (double*)malloc(n * local_n * sizeof(double));
    double* local_C = (double*)calloc(local_n * local_n, sizeof(double));
    
    // Distribute data using MPI_Scatter
    MPI_Scatter(mat1, local_n * n, MPI_DOUBLE, local_A, local_n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(mat2, n * local_n, MPI_DOUBLE, local_B, n * local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Start timing
//    double start_time = MPI_Wtime();
    
    // Perform the Ring algorithm for matrix multiplication
    for (int k = 0; k < n; k++) {
        // Broadcast the k-th row of A and k-th column of B
        MPI_Bcast(&local_A[k * local_n], local_n, MPI_DOUBLE, (rank + k) % size, MPI_COMM_WORLD);
        MPI_Bcast(&local_B[k], local_n, MPI_DOUBLE, (rank + k) % size, MPI_COMM_WORLD);

        // Local matrix multiplication
        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < local_n; j++) {
                for (int l = 0; l < n; l++) {
                    local_C[i * local_n + j] += local_A[i * n + l] * local_B[l * local_n + j];
                }
            }
        }
    }

    // End timing
//    double end_time = MPI_Wtime();

    // Gather results using MPI_Gather
    MPI_Gather(local_C, local_n * local_n, MPI_DOUBLE, mat3, local_n * local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

 // Clean up
    free(local_A);
    free(local_B);
    free(local_C);

    // Print timing information on rank 0
//    if (rank == 0) {
//    printf("Time spent for multiplying two %d by %d matrices with %d processes is %f seconds.\n", n, n, size, end_time - start_time);
//    }
}

int main( int argc, char *argv[] )
{
    int rank, size;
    int n1, n2, n3;
    double* mat1;
    double* mat2;
    double* mat3;
    double* matrix1;
    double* matrix2;
    double* matrix3;
    double* result1;
    double* result1_parallel;
    double* result2;
    double* result3;
    double start_time1, end_time1;
    double start_time2, end_time2;

    n1 = pow(2, 12);
    n2 = pow(2, 8);
    n3 = pow(2, 10);
    mat1 = (double *)malloc(n1 * n1 * sizeof(double));
    mat2 = (double *)malloc(n2 * n2 * sizeof(double));
    mat3 = (double *)malloc(n3 * n3 * sizeof(double));
    matrix1 = (double *)malloc(n1 * n1 * sizeof(double));
    matrix2 = (double *)malloc(n2 * n2 * sizeof(double));
    matrix3 = (double *)malloc(n3 * n3 * sizeof(double));
    result1 = (double *)malloc(n1 * n1 * sizeof(double));
    result1_parallel = (double *)malloc(n1 * n1 * sizeof(double));
    result2 = (double *)malloc(n2 * n2 * sizeof(double));
    result3 = (double *)malloc(n3 * n3 * sizeof(double));

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

//    if(rank == 0){
    //fill the matrices with random double precision floating point numbers between -1 and 1
    for (int i=0; i<n1; i++){
        for (int j=0; j<n1; j++){
            mat1[i*n1+j] = (double)rand()/RAND_MAX*2.0-1.0;
            matrix1[i*n1+j] = mat1[i*n1+j];
        }
    }
/*    for (int i=0; i<n2; i++){
        for (int j=0; j<n2; j++){
            mat2[i*n2+j] = (double)rand()/RAND_MAX*2.0-1.0;
            matrix2[i*n2+j] = mat2[i*n2+j];
        }
    }    
    for (int i=0; i<n3; i++){
        for (int j=0; j<n3; j++){
            mat3[i*n3+j] = (double)rand()/RAND_MAX*2.0-1.0;
            matrix3[i*n3+j] = mat3[i*n3+j];
        }
    }
*/
    //multiply mat1 with matrix1 and store the result in result1 and measure the time
    MPI_Barrier(MPI_COMM_WORLD);
    start_time1 = MPI_Wtime();
    seq_mat_mult(mat1, matrix1, result1, n1);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time1 = MPI_Wtime();
    double time_spent1 = end_time1 - start_time1;
    printf("Time spent for sequential multiplying two %d by %d matrices is %f seconds.\n From rank: %d \n", n1, n1, time_spent1, rank);
    //multiply mat2 with matrix2 and store the result in result2 and measure the time
/*    clock_t begin2 = clock();
    seq_mat_mult(mat2, matrix2, result2, n2);
    clock_t end2 = clock();
    double time_spent2 = (double)(end2 - begin2) / CLOCKS_PER_SEC;
    printf("Time spent for multiplying two %d by %d matrices is %f seconds.\n", n2, n2, time_spent2);
    //multiply mat3 with matrix3 and store the result in result3 and measure the time
    clock_t begin3 = clock();
    seq_mat_mult(mat3, matrix3, result3, n3);
    clock_t end3 = clock();
    double time_spent3 = (double)(end3 - begin3) / CLOCKS_PER_SEC;
    printf("Time spent for multiplying two %d by %d matrices is %f seconds.\n", n3, n3, time_spent3);
*/


    MPI_Barrier(MPI_COMM_WORLD);
    start_time2 = MPI_Wtime();
    parallel_mat_mult(mat1, matrix1, result1, n1, size);
    MPI_Barrier(MPI_COMM_WORLD);
    end_time2 = MPI_Wtime();
    double time_spent2 = end_time1 - start_time1;
    printf("Time spent for parallel multiplying two %d by %d matrices is %f seconds.\n From rank: %d \n", n1, n1, time_spent1, rank);

    //print the matrices from mat mat mulip 1 into files for checking
    FILE *fp;   //file pointer
    fp = fopen("mat1.txt", "w");
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n1; j++)
            fprintf(fp, "%f ", mat1[i*n1+j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    fp = fopen("matrix1.txt", "w");
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n1; j++)
            fprintf(fp, "%f ", matrix1[i*n1+j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    fp = fopen("result1.txt", "w");
    for (int i=0; i<n1; i++)
    {
        for (int j=0; j<n1; j++)
            fprintf(fp, "%f ", result1[i*n1+j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
//    }

    printf("Hello from %d of %d processors.\n", rank, size);
    MPI_Finalize();

    //free the memory
    free(mat1);
    free(mat2);
    free(mat3);
    free(matrix1);
    free(matrix2);
    free(matrix3);
    free(result1);
    free(result2);
    free(result3);

    return 0;
}
