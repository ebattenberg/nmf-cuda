#include<stdio.h>
#include<stdlib.h>
#include"matrix.h"
#include<time.h>
#include<sys/time.h>

// status printed and convergence check every ITER_CHECK iterations
#define ITER_CHECK 25	
// max number of iterations
#define MAX_ITER 200 	
// set to zero to guarantee MAX_ITER iterations, 0.001 is a good value otherwise
#define CONVERGE_THRESH 0

// number of timers used in profiling (don't change)
#define TIMERS 10  
char *tname[] = {"total","sgemm","eps","vecdiv","vecmult","sumrows","sumcols","coldiv","rowdiv","check"};


void update_div(matrix W, matrix H, matrix X, const float thresh, const int max_iter, double* t, int verbose);
double get_time();
unsigned nextpow2(unsigned x);


int main(int argc, char *argv[]){


    //factor X into W*H
    matrix W,H,X;


    // read in matrix data:
    // X - matrix to factorize
    // W - initial W matrix
    // H - initial H matrix
    read_matrix(&W,"../W.bin");
    read_matrix(&X,"../X.bin");
    read_matrix(&H,"../H.bin");

    //make sure no zero elements
    matrix_eps(X);
    matrix_eps(H);
    matrix_eps(W);

    int max_iter;
    if(argc > 1)
        max_iter = atoi(argv[1]);
    else 
        max_iter = MAX_ITER;

    // iterative nmf minimization
    update_div(W,H,X,CONVERGE_THRESH,max_iter,NULL,1);


    // write results matrices to binary files
    // (can be read with export_bin.m in Matlab)
    write_matrix(W,"../Wout.bin");
    write_matrix(H,"../Hout.bin");

    destroy_matrix(&W);
    destroy_matrix(&H);
    destroy_matrix(&X);
    return 0;
}



double get_time(){
    //output time in microseconds

    //the following line is required for function-wise timing to work,
    //but it slows down overall execution time.
    //comment out for faster execution
    cudaThreadSynchronize(); 

    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)(t.tv_sec+t.tv_usec/1E6);
}

int start_time(double* t, int i)
{
    if (t != NULL)
    {
        t[i] -= get_time();
        return 1;
    }
    else 
        return 0;
}

int stop_time(double* t, int i)
{
    if (t != NULL)
    {
        t[i] += get_time();
        return 1;
    }
    else 
        return 0;
}



void update_div(matrix W0, matrix H0, matrix X0, const float thresh, const int max_iter, double *t,int verbose){
    //run iterative multiplicative updates on W,H

    cublasInit();

    const int M = W0.dim[0];
    const int K = W0.dim[1];
    const int N = H0.dim[1];

    // pad matrix dimensions to multiples of:
    const int PAD_MULT = 32;

    int M_padded = M;
    if (M%PAD_MULT != 0)
        M_padded = M + (PAD_MULT - (M % PAD_MULT));

    int K_padded = K;
    if (K%PAD_MULT != 0)
        K_padded = K + (PAD_MULT - (K % PAD_MULT));

    int N_padded = N;
    if (N%PAD_MULT != 0)
        N_padded = N + (PAD_MULT - (N % PAD_MULT));

    //unpadded test
    //M_padded = M;
    //N_padded = N;
    //K_padded = K;

    // find reduction parameters
    int MN_params[4] = {1,1,1,1}; //M*N size reduction (whole matrix)
    int N_params[4] = {1,1,1,1}; //N size reductions (rows)
    int M_params[4] = {1,1,1,1}; //M size reductions (cols)

    int rem;
    rem = nextpow2(N_padded/128 + (!(N_padded%128)?0:1));
    if (rem <= 128)
    {
        N_params[0] = 128;
        N_params[1] = rem;
    }
    else if (rem <= 512)
    {
        N_params[0] = rem;
        N_params[1] = 128;
    }
    else
    {
        fprintf(stderr,"reduction parameter error\n");
        exit(1);
    }


    rem = nextpow2(M_padded/128 + (!(M_padded%128)?0:1));
    if (rem <= 128)
    {
        M_params[0] = 128;
        M_params[1] = rem;
    }
    else if (rem <= 512)
    {
        M_params[0] = rem;
        M_params[1] = 128;
    }
    else
    {
        fprintf(stderr,"reduction parameter error\n");
        exit(1);
    }

    MN_params[0] = M_params[0];
    MN_params[1] = M_params[1];
    MN_params[2] = N_params[0];
    MN_params[3] = N_params[1];

    //printf("reduction parameters: ");
    //printf("%u,%u,%u,%u\n",MN_params[0],MN_params[1],MN_params[2],MN_params[3]);


    // block size in vector arithmetic operations
    const int BLOCK_SIZE = 128;





    //copy host matrices to device memory
    copy_matrix_to_device(&W0);
    copy_matrix_to_device(&H0);
    copy_matrix_to_device(&X0);


    //matrix to hold W*H
    matrix WH0;
    create_matrix_on_device(&WH0,M,N,0.0);


    int i;

    /*
       double t_array[TIMERS];
       if(t==NULL)
       t = t_array;
       */
    if (t != NULL)
    {
        for(i=0;i<TIMERS;i++)
            t[i] = 0;
    }

    //float nancheck, zerocheck;
    // compute initial divergence and error
    float diff,div,change,prev_diff,prev_div;
    matrix_multiply_d(W0,H0,WH0);
    diff = matrix_difference_norm_d(compute,X0,WH0,MN_params);


    div = matrix_div_d(compute,X0,WH0,MN_params);
    if(verbose)
        printf("i: %4i, error: %6.4f, initial div: %8.4e\n",0,diff,div);


    // free device memory for unpadded matrices
    free_matrix_on_device(&W0);
    free_matrix_on_device(&H0);
    free_matrix_on_device(&X0);
    free_matrix_on_device(&WH0);


    //initialize temp matrices -----------------------


    //matrix to hold X./(W*H+EPS)
    matrix Z;
    create_matrix_on_device(&Z,M_padded,N_padded,0.0);

    //matrix to hold W'*Z
    matrix WtZ;
    create_matrix_on_device(&WtZ,K_padded,N_padded,0.0);

    //matrix to hold Z*H'
    matrix ZHt;
    create_matrix_on_device(&ZHt,M_padded,K_padded,0.0);

    //matrix to hold sum(W) [sum of cols of W]
    matrix sumW;
    create_matrix_on_device(&sumW,1,K_padded,0.0);

    //matrix to hold sum(H,2) [sum of rows of H]
    matrix sumH2;
    create_matrix_on_device(&sumH2,K_padded,1,0.0);


    //matrices to hold padded versions of matrices
    matrix W;
    create_matrix_on_device(&W,M_padded,K_padded,0.0);

    matrix H;
    create_matrix_on_device(&H,K_padded,N_padded,0.0);

    matrix X;
    create_matrix_on_device(&X,M_padded,N_padded,0.0);




    // move host matrices to padded device memory
    copy_matrix_to_device_padded(W0,W);
    copy_matrix_to_device_padded(H0,H);
    copy_matrix_to_device_padded(X0,X);




    //t[0] -= get_time();
    start_time(t,0);

        //matrix test1;

        for(i=0;i<max_iter;i++){

            //check for convergence, print status
            if(i % ITER_CHECK == 0 && i != 0){
                //t[9] -= get_time();
                start_time(t,9);
                matrix_multiply_d(W,H,Z);
                prev_diff = diff;
                diff = matrix_difference_norm_d(compute,X,Z,MN_params);
                change = (prev_diff-diff)/prev_diff;
                //t[9] += get_time();
                stop_time(t,9);
                if(verbose)
                    printf("i: %4i, error: %6.4f, %% change: %8.5f\n",
                            i,diff,change);
                if(change < thresh){
                    printf("converged\n");
                    break;
                }
            }


            /* matlab algorithm
               Z = X./(W*H+eps); H = H.*(W'*Z)./(repmat(sum(W)',1,F)); 
               Z = X./(W*H+eps);
               W = W.*(Z*H')./(repmat(sum(H,2)',N,1));
               */

            //
            // UPDATE H -----------------------------
            //


            //WH = W*H
            //t[1] -= get_time();
            start_time(t,1);
            matrix_multiply_d(W,H,Z);
            //t[1] += get_time();
            stop_time(t,1);




            //WH = WH+EPS
            //t[2] -= get_time();
            start_time(t,2);
            matrix_eps_d(Z,BLOCK_SIZE);
            //t[2] += get_time();
            stop_time(t,2);


            //Z = X./WH
            //t[3] -= get_time();
            start_time(t,3);
            element_divide_d(X,Z,Z,BLOCK_SIZE);
            //t[3] += get_time();
            stop_time(t,3);


            //sum cols of W into row vector
            //t[6] -= get_time();
            start_time(t,6);
            sum_cols_d(compute,W,sumW,M_params);
            matrix_eps_d(sumW,32);
            //t[6] += get_time();
            stop_time(t,6);

            //convert sumW to col vector (transpose)
            sumW.dim[0] = sumW.dim[1];
            sumW.dim[1] = 1;


            //WtZ = W'*Z
            //t[1] -= get_time();
            start_time(t,1);
            matrix_multiply_AtB_d(W,Z,WtZ);
            //t[1] += get_time();
            stop_time(t,1);


            //WtZ = WtZ./(repmat(sum(W)',1,H.dim[1])
            //[element divide cols of WtZ by sumW']
            //t[7] -= get_time();
            start_time(t,7);
            col_divide_d(WtZ,sumW,WtZ);
            //t[7] += get_time();
            stop_time(t,7);



            //H = H.*WtZ
            //t[4] -= get_time();
            start_time(t,4);
            element_multiply_d(H,WtZ,H,BLOCK_SIZE);
            //t[4] += get_time();
            stop_time(t,4);



            //
            // UPDATE W ---------------------------
            //

            //WH = W*H
            //t[1] -= get_time();
            start_time(t,1);
            matrix_multiply_d(W,H,Z);
            //t[1] += get_time();
            stop_time(t,1);


            //WH = WH+EPS
            //t[2] -= get_time();
            start_time(t,2);
            matrix_eps_d(Z,BLOCK_SIZE);
            //t[2] += get_time();
            stop_time(t,2);

            //Z = X./WH
            //t[3] -= get_time();
            start_time(t,3);
            element_divide_d(X,Z,Z,BLOCK_SIZE);
            //t[3] += get_time();
            stop_time(t,3);


            //sum rows of H into col vector
            //t[5] -= get_time();
            start_time(t,5);
            sum_rows_d(compute,H,sumH2,N_params);
            matrix_eps_d(sumH2,32);
            //t[5] += get_time();
            stop_time(t,5);

            //convert sumH2 to row vector (transpose)
            sumH2.dim[1] = sumH2.dim[0];
            sumH2.dim[0] = 1;

            //ZHt = Z*H'
            //t[1] -= get_time();
            start_time(t,1);
            matrix_multiply_ABt_d(Z,H,ZHt);
            //t[1] += get_time();
            stop_time(t,1);

            //ZHt = ZHt./(repmat(sum(H,2)',W.dim[0],1)
            //[element divide rows of ZHt by sumH2']
            //t[8] -= get_time();
            start_time(t,8);
            row_divide_d(ZHt,sumH2,ZHt);
            //t[8] += get_time();
            stop_time(t,8);

            //W = W.*ZHt
            //t[4] -= get_time();
            start_time(t,4);
            element_multiply_d(W,ZHt,W,BLOCK_SIZE);
            //t[4] += get_time();
            stop_time(t,4);


            // ------------------------------------

            //reset sumW to row vector
            sumW.dim[1] = sumW.dim[0];
            sumW.dim[0] = 1;
            //reset sumH2 to col vector
            sumH2.dim[0] = sumH2.dim[1];
            sumH2.dim[1] = 1;

            // ---------------------------------------

        }

    //t[0] += get_time();
    stop_time(t,0);




    //reallocate unpadded device memory
    allocate_matrix_on_device(&W0);
    allocate_matrix_on_device(&H0);

    //copy padded matrix to unpadded matrices
    copy_from_padded(W0,W);
    copy_from_padded(H0,H);

    // free padded matrices
    destroy_matrix(&W);
    destroy_matrix(&H);
    destroy_matrix(&X);

    // free temp matrices
    destroy_matrix(&Z);
    destroy_matrix(&WtZ);
    destroy_matrix(&ZHt);
    destroy_matrix(&sumW);
    destroy_matrix(&sumH2);

    copy_matrix_to_device(&X0);
    create_matrix_on_device(&WH0,M,N,0.0);

    // copy device results to host memory
    copy_matrix_from_device(&W0);
    copy_matrix_from_device(&H0);

    // evaluate final results
    matrix_multiply_d(W0,H0,WH0);
    prev_diff = diff;
    diff = matrix_difference_norm_d(compute,X0,WH0,MN_params);
    prev_div = div;
    div = matrix_div_d(compute,X0,WH0,MN_params);
    if(verbose){
        change = (prev_diff-diff)/prev_diff;
        printf("max iterations reached\n");
        printf("i: %4i, error: %6.4f, %% change: %8.5f\n",
                i,diff,change);
        change = (prev_div-div)/prev_div;
        printf("\tfinal div: %8.4e, %% div change: %8.5f\n",
                div,change);

        printf("\n");
        if (t != NULL)
        {
            for(i=0;i<TIMERS;i++)
                printf("t[%i]: %8.3f (%6.2f %%) %s\n",i,t[i],t[i]/t[0]*100,tname[i]);
        }
    }

    //clean up extra reduction memory
    matrix_difference_norm_d(cleanup,X0,WH0,MN_params);
    matrix_div_d(cleanup,X0,WH0,MN_params);
    sum_cols_d(cleanup,W,sumW,M_params);
    sum_rows_d(cleanup,H,sumH2,N_params);

    // free device memory for unpadded matrices
    free_matrix_on_device(&W0);
    free_matrix_on_device(&H0);
    free_matrix_on_device(&X0);

    // free temp matrices
    destroy_matrix(&WH0);

    cublasShutdown();

}

unsigned nextpow2(unsigned x) 
{
    x = x - 1;
    x = x | (x >> 1);
    x = x | (x >> 2);
    x = x | (x >> 4);
    x = x | (x >> 8);
    x = x | (x >> 16);
    return x + 1;

}
