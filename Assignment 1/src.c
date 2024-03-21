#include <mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// #pragma prutor-mpi-args: -np 12 -ppn 6

// #pragma prutor-mpi-sysargs: 4 4194304 10 7 5
// #pragma prutor-mpi-sysargs: 4 4 1 7 5

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    MPI_Status status;

    int my_rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int Px, Py;
    Px = atoi(argv[1]);
    Py = size/Px;
    
    int data_points = atoi(argv[2]);
    int side_len = sqrt(data_points);
    int time_steps = atoi(argv[3]);
    int seed = atoi(argv[4]);
    int stencil = atoi(argv[5]);
    
    srand(seed*(my_rank+10));
    
    double** data = (double**)malloc(side_len*sizeof(double*));
    double** final = (double**)malloc(side_len*sizeof(double*));
    
    for(int i=0; i<side_len; i++){
        data[i] = (double*)malloc(side_len*sizeof(double));
        final[i] = (double*)malloc(side_len*sizeof(double));
        for(int j=0; j<side_len; j++){
            data[i][j]= abs(rand()+(i*rand()+j*my_rank))/100;
            data[i][j] = 1;
        }
    }

    if(stencil==5){
        double up_recv[side_len], down_recv[side_len], left_recv[side_len], right_recv[side_len];  // Received from up, down, left, right
        double left_send[side_len], right_send[side_len];  // Sent to up, down, left, right
        
        double stime = MPI_Wtime();
        for(int t=0; t<time_steps; t++){
            
            // Pack for the columns and send for the rows directly
            for(int i=0; i<side_len; ++i){
                for(int j=0; j<side_len; ++j){
                    
                    final[i][j] = data[i][j];
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
                    MPI_Unpack(left_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
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
                    MPI_Unpack(left_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                }
                MPI_Send(left_send, side_len*8, MPI_PACKED, my_rank-1, my_rank-1, MPI_COMM_WORLD);
            }

            // Intra-Columns NN-2
            int col_rank = my_rank/Px;
            if(col_rank%2==0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data[side_len-1], side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[i];
                }
            }
            else if(col_rank%2!=0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[i];
                }
                MPI_Send(data[0], side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }
            if(col_rank%2!=0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data[side_len-1], side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[i];
                }
            }
            else if(col_rank%2==0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[i];
                }
                MPI_Send(data[0], side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }
            
            int Pi=my_rank/4, Pj=my_rank%4;

            for(int i=0; i<side_len; ++i){
                for(int j=0; j<side_len; ++j){
                    int denom = 5;
                    if(Pi == 0 &&  i==0)
                        denom--;
                    if(Pj == 0 && j==0)
                        denom--;
                    if(Pi == (Py-1) && (i == (side_len-1)))
                        denom--;
                    if(Pj == (Px-1) && (j == (side_len-1)))
                        denom--;
                    final[i][j] /= denom;
                    data[i][j] = final[i][j];
                }
            }
        }

        double etime = MPI_Wtime();
        double ctime = etime - stime;
        double max_time;
        MPI_Reduce(&ctime, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(my_rank==0){
            printf("Max time: %lf\n", max_time);
        }

    }
    else if(stencil==9){

        double up_recv[2*side_len], down_recv[2*side_len], left_recv[2*side_len], right_recv[2*side_len];  // Received from up, down, left, right
        double left_send[2*side_len], right_send[2*side_len];  // Sent to up, down, left, right

        int num_elements = side_len; // Number of elements to pack
        
        double** temp = (double**)malloc(side_len*sizeof(double*));

        double stime = MPI_Wtime();

        for(int t=0; t<time_steps; t++){
            // Pack for the columns and send for the rows directly
            
            for(int i=0; i<side_len; ++i){
                for(int j=0; j<side_len; ++j){
                    final[i][j] = data[i][j];
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
                    if(i>1){
                        final[i][j] += data[i-2][j];
                    }
                    if(i<side_len-2){
                        final[i][j] += data[i+2][j];
                    }
                    if(j>1){
                        final[i][j] += data[i][j-2];
                    }
                    if(j<side_len-2){
                        final[i][j] += data[i][j+2];
                    }
                }
            }

            int position;
            position = 0;
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
            }
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data[i][1], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
            }
            position = 0;
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
            }
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data[i][side_len-2], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
            }

            // Send using NN-2 first for all rows, then for all columns

            // Intra-Rows NN-2
            if(my_rank%2==0 && (my_rank+1)%Px!=0){  // Right send/recv
                MPI_Send(right_send, 2*side_len*8, MPI_PACKED, my_rank+1, my_rank+1, MPI_COMM_WORLD);
                MPI_Recv(right_recv, 2*side_len*8, MPI_PACKED, my_rank+1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len-1] += temp;
                    final[i][side_len-2] += temp;
                }
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len-1] += temp;
                }
            }
            else if(my_rank%2!=0 && (my_rank)%Px>0){  // Left send/recv
            
                MPI_Recv(left_recv, 2*side_len*8, MPI_PACKED, my_rank-1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(left_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                    final[i][1] += temp;
                }
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(left_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                }
                MPI_Send(left_send, 2*side_len*8, MPI_PACKED, my_rank-1, my_rank-1, MPI_COMM_WORLD);
            }

            if(my_rank%2!=0 && (my_rank+1)%Px!=0){  // Right send/recv
                MPI_Send(right_send, 2*side_len*8, MPI_PACKED, my_rank+1, my_rank+1, MPI_COMM_WORLD);
                MPI_Recv(right_recv, 2*side_len*8, MPI_PACKED, my_rank+1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len-1] += temp;
                    final[i][side_len-2] += temp;
                }
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(right_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len-1] += temp;
                }
            }
            else if(my_rank%2==0 && (my_rank)%Px!=0){  // Left send/recv
                MPI_Recv(left_recv, 2*side_len*8, MPI_PACKED, my_rank-1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(left_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                    final[i][1] += temp;
                }
                for(int i=0; i<side_len; i++){
                    MPI_Unpack(left_recv, side_len*8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                }
                MPI_Send(left_send, 2*side_len*8, MPI_PACKED, my_rank-1, my_rank-1, MPI_COMM_WORLD);
            }

            // Intra-Columns NN-2
            int col_rank = my_rank/Px;
            if(col_rank%2==0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data[side_len-2], 2*side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, 2*side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[i];
                    final[side_len-2][i] += down_recv[i];
                }
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[i];
                }
            }
            else if(col_rank%2!=0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, 2*side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[i];
                }
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[i];
                    final[1][i] += up_recv[i];
                }
                MPI_Send(data[0], 2*side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }
            if(col_rank%2!=0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data[side_len-2], 2*side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, 2*side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[i];
                    final[side_len-2][i] += down_recv[i];
                }
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[i];
                }
            }
            else if(col_rank%2==0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, 2*side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[i];
                }
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[i];
                    final[1][i] += up_recv[i];
                }
                MPI_Send(data[0], 2*side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }

            int Pi=my_rank/4, Pj=my_rank%4;
            for(int i=0; i<side_len; ++i){
                for(int j=0; j<side_len; ++j){
                    int denom = 9;
                    if(Pi == 0) {
                        if(i == 0)
                            denom -= 2;
                        else if (i == 1)
                            denom -= 1;
                    }
                    if(Pj == 0) {
                        if(j == 0)
                            denom -= 2;
                        else if (j == 1)
                            denom -= 1;
                    }
                    if(Pi == (Py - 1)) {
                        if(i == (side_len-1))
                            denom -= 2;
                        else if  (i == (side_len-2))
                            denom -= 1;
                    }
                    if(Pj == (Px - 1)) {
                        if(j == (side_len-1))
                            denom -= 2;
                        else if (j == (side_len-2))
                            denom -= 1;
                    }
                    final[i][j] /= denom;
                    data[i][j] = final[i][j];
                }
            }
        }

        double etime = MPI_Wtime();
        double time = etime - stime;
        double max_time;
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(my_rank==0){
            printf("Max time: %lf\n", max_time);
        }
    }    
    // printf("Complete\n");
    MPI_Finalize();
}