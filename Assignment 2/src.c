#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// #pragma prutor-mpi-args: -np 12 -ppn 4

// #pragma prutor-mpi-sysargs: 4 4 1 7 

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    MPI_Status status;

    int my_rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int Px, Py;
    Px = atoi(argv[1]);
    Py = size / Px;

    MPI_Comm newcomm;
    int color = my_rank / Px;
    MPI_Comm_split (MPI_COMM_WORLD, color, my_rank, &newcomm);

    int data_points = atoi(argv[2]);
    int side_len = sqrt(data_points);
    int time_steps = atoi(argv[3]);
    int seed = atoi(argv[4]);

    srand(seed * (my_rank + 10));

    double **data = (double **)malloc(side_len * sizeof(double *));
    double **data_without_leader = (double **)malloc(side_len * sizeof(double *));
    double **final = (double **)malloc(side_len * sizeof(double *));

    for (int i = 0; i < side_len; i++)
    {
        data[i] = (double *)malloc(side_len * sizeof(double));
        data_without_leader[i] = (double *)malloc(side_len * sizeof(double));
        final[i] = (double *)malloc(side_len * sizeof(double));
        for (int j = 0; j < side_len; j++)
        {
            data[i][j] = abs(rand() + (i * rand() + j * my_rank));
            data_without_leader[i][j] = data[i][j];
        }
    }

    int leader = 1;
    /******************* LEADER RANK  ***************************/
    if(leader == 1) {
        double inter_node_up_recv[2 * Px * side_len], inter_node_down_recv[2 * Px * side_len], left_recv[2 * side_len], right_recv[2 * side_len]; // Received from up, down, left, right
        double left_send[2 * side_len], right_send[2 * side_len]; // Sent to up, down, left, right
        double intra_node_left_recv_down[2 * Px * side_len], intra_node_left_recv_up[2 * Px * side_len];
        double up_recv[2 * side_len], down_recv[2 * side_len];

        int num_elements = side_len; // Number of elements to pack

        double **temp = (double **)malloc(side_len * sizeof(double *));

        double stime = MPI_Wtime();

        for (int t = 0; t < time_steps; t++)
        {
            // Pack for the columns and send for the rows directly

            for (int i = 0; i < side_len; ++i)
            {
                for (int j = 0; j < side_len; ++j)
                {
                    final[i][j] = data[i][j];
                    if (i)
                    {
                        final[i][j] += data[i - 1][j];
                    }
                    if (i < side_len - 1)
                    {
                        final[i][j] += data[i + 1][j];
                    }
                    if (j)
                    {
                        final[i][j] += data[i][j - 1];
                    }
                    if (j < side_len - 1)
                    {
                        final[i][j] += data[i][j + 1];
                    }
                    if (i > 1)
                    {
                        final[i][j] += data[i - 2][j];
                    }
                    if (i < side_len - 2)
                    {
                        final[i][j] += data[i + 2][j];
                    }
                    if (j > 1)
                    {
                        final[i][j] += data[i][j - 2];
                    }
                    if (j < side_len - 2)
                    {
                        final[i][j] += data[i][j + 2];
                    }
                }
            }

            int position;
            position = 0;
            for (int i = 0; i < side_len; i++)
            {
                MPI_Pack(&data[i][0], 1, MPI_DOUBLE, left_send, side_len * 8, &position, MPI_COMM_WORLD);
            }
            for (int i = 0; i < side_len; i++)
            {
                MPI_Pack(&data[i][1], 1, MPI_DOUBLE, left_send, side_len * 8, &position, MPI_COMM_WORLD);
            }
            position = 0;
            for (int i = 0; i < side_len; i++)
            {
                MPI_Pack(&data[i][side_len - 1], 1, MPI_DOUBLE, right_send, side_len * 8, &position, MPI_COMM_WORLD);
            }
            for (int i = 0; i < side_len; i++)
            {
                MPI_Pack(&data[i][side_len - 2], 1, MPI_DOUBLE, right_send, side_len * 8, &position, MPI_COMM_WORLD);
            }

            // Send using NN-2 first for all rows, then for all columns

            // Intra-Rows NN-2
            if (my_rank % 2 == 0 && (my_rank + 1) % Px != 0)
            { // Right send/recv
                MPI_Send(right_send, 2 * side_len * 8, MPI_PACKED, my_rank + 1, my_rank + 1, MPI_COMM_WORLD);
                MPI_Recv(right_recv, 2 * side_len * 8, MPI_PACKED, my_rank + 1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for (int i = 0; i < side_len; i++)
                {
                    MPI_Unpack(right_recv, side_len * 8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len - 1] += temp;
                    final[i][side_len - 2] += temp;
                }
                for (int i = 0; i < side_len; i++)
                {
                    MPI_Unpack(right_recv, side_len * 8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len - 1] += temp;
                }
            }
            else if (my_rank % 2 != 0 && (my_rank) % Px > 0)
            { // Left send/recv

                MPI_Recv(left_recv, 2 * side_len * 8, MPI_PACKED, my_rank - 1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for (int i = 0; i < side_len; i++)
                {
                    MPI_Unpack(left_recv, side_len * 8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                    final[i][1] += temp;
                }
                for (int i = 0; i < side_len; i++)
                {
                    MPI_Unpack(left_recv, side_len * 8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                }
                MPI_Send(left_send, 2 * side_len * 8, MPI_PACKED, my_rank - 1, my_rank - 1, MPI_COMM_WORLD);
            }

            if (my_rank % 2 != 0 && (my_rank + 1) % Px != 0)
            { // Right send/recv
                MPI_Send(right_send, 2 * side_len * 8, MPI_PACKED, my_rank + 1, my_rank + 1, MPI_COMM_WORLD);
                MPI_Recv(right_recv, 2 * side_len * 8, MPI_PACKED, my_rank + 1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for (int i = 0; i < side_len; i++)
                {
                    MPI_Unpack(right_recv, side_len * 8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len - 1] += temp;
                    final[i][side_len - 2] += temp;
                }
                for (int i = 0; i < side_len; i++)
                {
                    MPI_Unpack(right_recv, side_len * 8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][side_len - 1] += temp;
                }
            }
            else if (my_rank % 2 == 0 && (my_rank) % Px != 0)
            { // Left send/recv
                MPI_Recv(left_recv, 2 * side_len * 8, MPI_PACKED, my_rank - 1, my_rank, MPI_COMM_WORLD, &status);
                position = 0;
                double temp;
                for (int i = 0; i < side_len; i++)
                {
                    MPI_Unpack(left_recv, side_len * 8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                    final[i][1] += temp;
                }
                for (int i = 0; i < side_len; i++)
                {
                    MPI_Unpack(left_recv, side_len * 8, &position, &temp, 1, MPI_DOUBLE, MPI_COMM_WORLD);
                    final[i][0] += temp;
                }
                MPI_Send(left_send, 2 * side_len * 8, MPI_PACKED, my_rank - 1, my_rank - 1, MPI_COMM_WORLD);
            }
                    
            MPI_Gather(&data[0][0], 2 * side_len, MPI_DOUBLE, intra_node_left_recv_up, 2 * side_len, MPI_DOUBLE, 0, newcomm);  // Gathers the 4 rows in the intra_node_left_recv buffer of the first column process
            MPI_Gather(&data[side_len-2][0], 2 * side_len, MPI_DOUBLE, intra_node_left_recv_down, 2 * side_len, MPI_DOUBLE, 0, newcomm);  // Gathers the 4 rows in the intra_node_left_recv buffer of the first column process

            // Intra-Columns NN-2
            if(my_rank % Px == 0) {
                int col_rank = my_rank / Px;
                if (col_rank % 2 == 0 && col_rank < Py - 1)
                { // Down send/recv
                    MPI_Send(intra_node_left_recv_down, 2 * Px * side_len, MPI_DOUBLE, my_rank + Px, my_rank + Px, MPI_COMM_WORLD);
                    MPI_Recv(inter_node_down_recv, 2 * Px * side_len, MPI_DOUBLE, my_rank + Px, my_rank, MPI_COMM_WORLD, &status);
                }
                else if (col_rank % 2 != 0 && col_rank > 0)
                { // Up send/recv
                    MPI_Recv(inter_node_up_recv, 2 * Px * side_len, MPI_DOUBLE, my_rank - Px, my_rank, MPI_COMM_WORLD, &status);
                    MPI_Send(intra_node_left_recv_up, 2 * Px * side_len, MPI_DOUBLE, my_rank - Px, my_rank - Px, MPI_COMM_WORLD);
                }
                if (col_rank % 2 != 0 && col_rank < Py - 1)
                { // Down send/recv
                    MPI_Send(intra_node_left_recv_down,2 * Px * side_len, MPI_DOUBLE, my_rank + Px, my_rank + Px, MPI_COMM_WORLD);
                    MPI_Recv(inter_node_down_recv, 2 * Px * side_len, MPI_DOUBLE, my_rank + Px, my_rank, MPI_COMM_WORLD, &status);
                }
                else if (col_rank % 2 == 0 && col_rank > 0)
                { // Up send/recv
                    MPI_Recv(inter_node_up_recv, 2 * Px * side_len, MPI_DOUBLE, my_rank - Px, my_rank, MPI_COMM_WORLD, &status);
                    MPI_Send(intra_node_left_recv_up, 2 * Px * side_len, MPI_DOUBLE, my_rank - Px, my_rank - Px, MPI_COMM_WORLD);
                }
            }
            MPI_Scatter(inter_node_up_recv, 2 * side_len, MPI_DOUBLE, up_recv, 2 * side_len, MPI_DOUBLE, 0, newcomm);
            MPI_Scatter(inter_node_down_recv, 2 * side_len, MPI_DOUBLE, down_recv, 2 * side_len, MPI_DOUBLE, 0, newcomm);
            if(my_rank / Px != 0){
                for (int i = 0; i < side_len; i++)
                {
                    final[0][i] += up_recv[i];
                    final[0][i] += up_recv[side_len + i];
                    final[1][i] += up_recv[side_len + i];
                }
            }
            if(my_rank / Px != Py - 1){
                for (int i = 0; i < side_len; i++)
                {
                    final[side_len - 1][i] += down_recv[i];
                    final[side_len - 1][i] += down_recv[side_len + i];
                    final[side_len - 2][i] += down_recv[i];
                }
            }

            int Pi = my_rank / Px, Pj = my_rank % Px;
            for (int i = 0; i < side_len; ++i)
            {
                for (int j = 0; j < side_len; ++j)
                {
                    int denom = 9;
                    if (Pi == 0)
                    {
                        if (i == 0)
                            denom -= 2;
                        else if (i == 1)
                            denom -= 1;
                    }
                    if (Pj == 0)
                    {
                        if (j == 0)
                            denom -= 2;
                        else if (j == 1)
                            denom -= 1;
                    }
                    if (Pi == (Py - 1))
                    {
                        if (i == (side_len - 1))
                            denom -= 2;
                        else if (i == (side_len - 2))
                            denom -= 1;
                    }
                    if (Pj == (Px - 1))
                    {
                        if (j == (side_len - 1))
                            denom -= 2;
                        else if (j == (side_len - 2))
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
        if (my_rank == 0)
        {
            printf("Time with leader = %lf\n", max_time);
        }
    }

    leader = 0;
    /******************* WITHOUT LEADER ***************************/
    if(leader == 0) {
        double up_recv[2*side_len], down_recv[2*side_len], left_recv[2*side_len], right_recv[2*side_len];  // Received from up, down, left, right
        double left_send[2*side_len], right_send[2*side_len];  // Sent to up, down, left, right

        int num_elements = side_len; // Number of elements to pack
        
        double** temp = (double**)malloc(side_len*sizeof(double*));

        double stime = MPI_Wtime();

        for(int t=0; t<time_steps; t++){
            // Pack for the columns and send for the rows directly
            
            for(int i=0; i<side_len; ++i){
                for(int j=0; j<side_len; ++j){
                    final[i][j] = data_without_leader[i][j];
                    if(i){
                        final[i][j] += data_without_leader[i-1][j];
                    }
                    if(i<side_len-1){
                        final[i][j] += data_without_leader[i+1][j];
                    }
                    if(j){
                        final[i][j] += data_without_leader[i][j-1];
                    }
                    if(j<side_len-1){
                        final[i][j] += data_without_leader[i][j+1];
                    }
                    if(i>1){
                        final[i][j] += data_without_leader[i-2][j];
                    }
                    if(i<side_len-2){
                        final[i][j] += data_without_leader[i+2][j];
                    }
                    if(j>1){
                        final[i][j] += data_without_leader[i][j-2];
                    }
                    if(j<side_len-2){
                        final[i][j] += data_without_leader[i][j+2];
                    }
                }
            }

            int position;
            position = 0;
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data_without_leader[i][0], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
            }
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data_without_leader[i][1], 1, MPI_DOUBLE, left_send, side_len*8, &position, MPI_COMM_WORLD);
            }
            position = 0;
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data_without_leader[i][side_len-1], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
            }
            for(int i = 0; i < side_len; i++){
                    MPI_Pack(&data_without_leader[i][side_len-2], 1, MPI_DOUBLE, right_send, side_len*8, &position, MPI_COMM_WORLD);
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
                MPI_Send(data_without_leader[side_len-2], 2*side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, 2*side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[i];
                    final[side_len-2][i] += down_recv[i];
                }
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[side_len + i];
                }
            }
            else if(col_rank%2!=0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, 2*side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[side_len + i];
                }
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[i];
                    final[1][i] += up_recv[i];
                }
                MPI_Send(data_without_leader[0], 2*side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
            }
            if(col_rank%2!=0 && col_rank<Py-1){  // Down send/recv
                MPI_Send(data_without_leader[side_len-2], 2*side_len, MPI_DOUBLE, my_rank+Px, my_rank+Px, MPI_COMM_WORLD);
                MPI_Recv(down_recv, 2*side_len, MPI_DOUBLE, my_rank+Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[i];
                    final[side_len-2][i] += down_recv[i];
                }
                for(int i=0; i<side_len; i++){
                    final[side_len-1][i] += down_recv[side_len + i];
                }
            }
            else if(col_rank%2==0 && col_rank>0){  // Up send/recv
                MPI_Recv(up_recv, 2*side_len, MPI_DOUBLE, my_rank-Px, my_rank, MPI_COMM_WORLD, &status);
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[side_len + i];
                }
                for(int i=0; i<side_len; i++){
                    final[0][i] += up_recv[i];
                    final[1][i] += up_recv[i];
                }
                MPI_Send(data_without_leader[0], 2*side_len, MPI_DOUBLE, my_rank-Px, my_rank-Px, MPI_COMM_WORLD);
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
                    data_without_leader[i][j] = final[i][j];
                }
            }
        }

        double etime = MPI_Wtime();
        double time = etime - stime;
        double max_time;
        MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if(my_rank==0){
            printf("TIme without leader = %lf\n", max_time);
        }
    }
    if(my_rank == 0) {
        printf("Data = %lf\n", data[0][0]);
    }
    
    MPI_Finalize();
}
