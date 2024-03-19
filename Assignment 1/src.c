#include <mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#pragma prutor-mpi-args: -np 12 -ppn 6

#pragma prutor-mpi-sysargs: 4 4194304 10 7 5



int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    MPI_Status status;

    int my_rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int Px, Py;
    Px = atoi(argv[1]);
    int data_points = atoi(argv[2]);
    int side_len = sqrt(data_points);
    Py = size/Px;
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

    if(stencil==5){
        // double up[Px], down[Px], left[Py], right[Py];
        double up_recv[side_len], down_recv[side_len], left_recv[side_len], right_recv[side_len];  // Received from up, down, left, right
        double left_send[side_len], right_send[side_len];  // Sent to up, down, left, right
        
        int toprow, bottomrow, leftcol, rightcol;
        int Pi=my_rank/4, Pj=my_rank%4;

        toprow = (Pi==0)?1:0;
        bottomrow = (Pi==Py-1)?1:0;
        leftcol = (Pj==0)?1:0;
        rightcol = (Pj==Px-1)?1:0;  // 1 if the process is at the edge of the grid

        int num_elements = side_len; // Number of elements to pack
        
        int stime = MPI_Wtime();

        double** temp = (double**)malloc(side_len*sizeof(double*));

        for(int t=0; t<time_steps; t++){
            // Pack for the columns and send for the rows directly
            
            for(int i=0; i<side_len; ++i){
                for(int j=0; j<side_len; ++j){
                    final[i][j] = 0;
                    if(i){
                        final[i][j] += data[i-1][j];
                    }
                    if(i<side_len-1){
                        final[i][j] += data[i+1][j];
                    }
                    if(j){
                        final[i][j] += data[i][j-1];
                    }
                    if(j<side_len-1){
                        final[i][j] += data[i][j+1];
                    }
                }
            }
            

            int position;
            position = 0;
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
            }
            position = 0;
            for(int i = 0; i < side_len; i++){
                MPI_Pack(&data[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
            }
            // printf("%d %d Pack done\n", my_rank, side_len);

            // Send using NN-2 first for all rows, then for all columns

            // Intra-Rows NN-2
            if(my_rank%2==0 && (my_rank+1)%Px!=0){  // Right send/recv
                MPI_Send(right_send, side_len*8, MPI_PACKED, my_rank+1, my_rank+1, MPI_COMM_WORLD);
                MPI_Recv(right_recv, side_len*8, MPI_PACKED, my_rank+1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len-1] += temp;
                }
            }
            else if(my_rank%2!=0 && (my_rank)%Px>0){  // Left send/recv
            
                MPI_Recv(left_recv, side_len*8, MPI_PACKED, my_rank-1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                }
                MPI_Send(left_send, side_len*8, MPI_PACKED, my_rank-1, my_rank-1, MPI_COMM_WORLD);
            }

            if(my_rank%2!=0 && (my_rank+1)%Px!=0){  // Right send/recv
                MPI_Send(right_send, side_len*8, MPI_PACKED, my_rank+1, my_rank+1, MPI_COMM_WORLD);
                MPI_Recv(right_recv, side_len*8, MPI_PACKED, my_rank+1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len-1] += temp;
                }
            }
            else if(my_rank%2==0 && (my_rank)%Px!=0){  // Left send/recv
                MPI_Recv(left_recv, side_len*8, MPI_PACKED, my_rank-1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                }
                MPI_Send(left_send, side_len*8, MPI_PACKED, my_rank-1, my_rank-1, MPI_COMM_WORLD);
            }
            
            

            // Intra-Columns NN-2
            int col_rank = my_rank/Px;
            if(col_rank%2==0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data[side_len-1], side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[side_len-1][i] += temp;
                }
            }
            else if(col_rank%2!=0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[0][i] += temp;
                }
                MPI_Send(data[0], side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }
            if(col_rank%2!=0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data[side_len-1], side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[side_len-1][i] += temp;
                }
            }
            else if(col_rank%2==0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[0][i] += temp;
                }
                MPI_Send(data[0], side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }

            temp = data;
            data = final;
            final = temp;

        }
    
    }
    else if(stencil==9){

    }
    
    printf("Complete\n");
    MPI_Finalize();
}