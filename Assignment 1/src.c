#include <mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

double** init_data(int n){
    double** ptr = (double**)malloc(n*sizeof(double*));

    for(int i=0; i<n; i++){
        ptr[i] = (double*)malloc(n*sizeof(double));
        for(int j=0; j<n; j++){
            srand(seed*(myrank+10));
            ptr[i][j]= abs(rand()+(i*rand()+j*myrank))/100;
        }
    }
    return ptr;
}

void compute_stencil(double** data, int n){
    for(int i=1; i<n-1; i++){
        for(int j=1; j<n-1; j++){
            data[i][j] = (data[i-1][j] + data[i+1][j] + data[i][j-1] + data[i][j+1] + data[i][j])/5;
        }
    }
}

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    MPI_Status status;

    int my_rank, p;
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int Px, Py;
    
    int data_points = atoi(argv[2]);
    int side_len = sqrt(data_points);

    Px = 4;
    Py = 3;
    int time_steps = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int stencil = atoi(argv[5]);

    double **data = init_data(side_len);


    if(stencil==5){
        // double up[Px], down[Px], left[Py], right[Py];
        double up_recv[Px], down_recv[Px], left_recv[Py], right_recv[Py];  // Received from up, down, left, right
        double up_send[Px], down_send[Px], left_send[Py], right_send[Py];  // Sent to up, down, left, right
        
        int toprow, bottomrow, leftcol, rightcol;
        int Pi=my_rank/4, Pj=my_rank%4;

        toprow = (Pi==0)?1:0;
        bottomrow = (Pi==Py-1)?1:0;
        leftcol = (Pj==0)?1:0;
        rightcol = (Pj==Px-1)?1:0;  // 1 if the process is at the edge of the grid

        int num_elements = side_len; // Number of elements to pack
        
        int stime = MPI_Wtime();

        for(int t=0; t<time_steps; t++){
            // Pack for the columns and send for the rows directly
            if(toprow){
                if(leftcol){
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
                    }
                }
                else if(rightcol){
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
                    }
                }
                else{
                    // Non-edge blocks
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
                        MPI_Pack(&data[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
                    }
                }
            }
            else if(bottomrow){
                if(leftcol){
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
                    }
                }
                else if(rightcol){
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
                    }
                }
                else{
                    // Non-edge blocks
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
                        MPI_Pack(&data[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
                    }
                }
            }
            else{
                if(leftcol){
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
                    }
                }
                else if(rightcol){
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
                    }
                }
                else{
                    // Non-edge blocks
                    int position = 0;
                    for(int i = 0; i < side_len; i++){
                        MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
                        MPI_Pack(&data[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
                    }

                }
            }

            // Send using NN-2 first for all rows, then for all columns

            // Inter-Rows NN-2
            if(my_rank%2==0 && (my_rank+1)%Px!=0){  // Right send/recv
                MPI_Send(right_send, side_len, MPI_DOUBLE, my_rank+1, my_rank+1, MPI_COMM_WORLD);
                MPI_Recv(right_recv, side_len, MPI_DOUBLE, my_rank+1, my_rank, MPI_COMM_WORLD, &status);
            }
            else if(my_rank%2!=0 && my_rank%Px!=0){  // Left send/recv
                MPI_Recv(left_recv, side_len, MPI_DOUBLE, my_rank-1, my_rank, MPI_COMM_WORLD, &status);
                MPI_Send(left_send, side_len, MPI_DOUBLE, my_rank-1, my_rank-1, MPI_COMM_WORLD);
            }

            if(my_rank%2!=0 && (my_rank+1)%Px!=0){  // Right send/recv
                MPI_Send(right_send, side_len, MPI_DOUBLE, my_rank+1, my_rank+1, MPI_COMM_WORLD);
                MPI_Recv(right_recv, side_len, MPI_DOUBLE, my_rank+1, my_rank, MPI_COMM_WORLD, &status);
            }
            else if(my_rank%2==0 && my_rank%Px!=0){  // Left send/recv
                MPI_Recv(left_recv, side_len, MPI_DOUBLE, my_rank-1, my_rank, MPI_COMM_WORLD, &status);
                MPI_Send(left_send, side_len, MPI_DOUBLE, my_rank-1, my_rank-1, MPI_COMM_WORLD);
            }

            // Inter-Columns NN-2
            int col_rank = my_rank/Px;
            if(col_rank%2==0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data[side_len-1], side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
            }
            else if(col_rank%2!=0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                MPI_Send(data[0], side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }
            if(col_rank%2!=0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data[side_len-1], side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
            }
            else if(col_rank%2==0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                MPI_Send(data[0], side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }

            

        }
    
    }
    else if(stencil==9){

    }
}