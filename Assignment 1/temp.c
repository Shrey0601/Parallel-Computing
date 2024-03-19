#include <mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#pragma prutor-mpi-args: -np 12 -ppn 6

#pragma prutor-mpi-sysargs: 4 262144 10 7 5



int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    MPI_Status status;

    int my_rank, p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int og_rank = my_rank;
    int Px, Py;
    Px = atoi(argv[1]);
    int data_points = atoi(argv[2]);
    int side_len = sqrt(data_points);
    Py = side_len/Px;
    int time_steps = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int stencil = atoi(argv[5]);
    double** data = (double**)malloc(side_len*sizeof(double*));
    for(int i=0; i<side_len; i++){
        data[i] = (double*)malloc(side_len*sizeof(double));
        for(int j=0; j<side_len; j++){
            srand(seed*(my_rank+10));
            data[i][j]= abs(rand()+(i*rand()+j*my_rank))/100;
        }
    }

    double** final = (double**)malloc(side_len*sizeof(double*));
    for(int i=0; i<side_len; i++){
        final[i] = (double*)malloc(side_len*sizeof(double));
    }
    
    free(data);
    free(final);
            
    
    printf("Complete");
}