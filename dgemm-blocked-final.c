/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#pragma GCC push_options
#pragma GCC optimize("unroll-loops")

#include <immintrin.h>
//#include <x86intrin.h>
#include <avx2intrin.h>
#include <stdint.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#include <string.h>
#include <stdlib.h>

/**
lv1 cache: 32k
lv2 cache: 256k
lv3 cache: 10k+k
**/


#if !defined(BLOCK_SIZE1)
//128 - 500
#define REGA 3
#define REGB 4 // B = 4*4
#define BLOCK_SIZE1 96
#define BLOCK_SIZE2 96

#endif


#define min(a,b) (((a)<(b))?(a):(b))



/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
__attribute__((optimize("unroll-loops")))
static void do_block (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int k = 0; k < K; ++k)
        {
            /* Compute C(i,j) */
            // register double cij = C[i*lda+j];
            double r = A[i * lda + k];
            for (int j = 0; j < N; ++j)
                C[i * lda + j] += r * B[k*lda+j];

            // C[i*lda+j] = cij;
        }
}



//M = REGA = 3, N = REGB*256/64 = 16
//for block1, M = 3, N = 16, which means all c00-c13 are stored in C
//K changeable
__attribute__((optimize("unroll-loops")))
static void do_block3_16(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C);
static void do_block2_16(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C);
static void do_block1_16(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C);




static inline void block_square_multilv0(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {
    for (int i = 0; i < M; i += REGA)
        for(int j = 0; j < N; j += REGB*4)
            for(int k = 0; k < K; k += BLOCK_SIZE1){
                int curM = min (REGA, M-i);
                int curN = min (REGB * 4, N-j);
                int curK = min (BLOCK_SIZE1, K-k);
                // int realloada = min(3, lda-N);
                // int realloadb = min(16, lda-M);
                if(curM == 3 && curN == REGB * 4)
                    do_block3_16(lda, curM, curN, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);
                else if(curN == 2 && curN == REGB * 4)
                    do_block2_16(lda, curM, curN, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);
                else if(curN == 1 && curN == REGB * 4)
                    do_block1_16(lda, curM, curN, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);
                else// not enough B
                    do_block(lda, curM, curN, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);

            }
}

#define BLOCK_SIZE_M 24
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32

static inline void block_square_multilv1(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C){ // 3*16
    for (int i = 0; i < M; i += BLOCK_SIZE_M)
        for(int j = 0; j < N; j += BLOCK_SIZE_N)
            for(int k = 0; k < K; k += BLOCK_SIZE_K){
                int curM = min (BLOCK_SIZE_M, M-i);
                int curN = min (BLOCK_SIZE_N, N-j);
                int curK = min (BLOCK_SIZE_K, K-k);

                block_square_multilv0(lda, curM, curN, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);

            }
}

// 96 x 96
static inline void block_square_multilv2(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C){
    for (int i = 0; i < M; i += BLOCK_SIZE2)
        for(int j = 0; j < N; j += BLOCK_SIZE2)
            for(int k = 0; k < K; k += BLOCK_SIZE2){
                int curM = min (BLOCK_SIZE2, M-i);
                int curN = min (BLOCK_SIZE2, N-j);
                int curK = min (BLOCK_SIZE2, K-k);

                block_square_multilv1(lda, curM, curN, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);

            }
}



void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
    // double __attribute__((align (32))) A_aligned[lda * lda];
    // double __attribute__((align (32))) B_aligned[lda * lda];
    // double __attribute__((align (__BIGGEST_ALIGNMENT__))) C_aligned[lda * lda];

    double* restrict A_aligned;
    double* restrict B_aligned;

    posix_memalign(&A_aligned, 32, lda * lda * 8);
    posix_memalign(&B_aligned, 32, lda * lda * 8);

    memcpy(A_aligned, A, lda * lda * 8);
    memcpy(B_aligned, B, lda * lda * 8);
    // memcpy(C_aligned, C, lda * lda * 8);

    block_square_multilv2(lda, lda, lda, lda, A_aligned, B_aligned, C);
    // memcpy(C, C_aligned, lda * lda * 8);
    free(A_aligned);
    free(B_aligned);
}





static inline void do_block3_16(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C){
    register __m256d c00,c01,c02,c03;
    register __m256d c10,c11,c12,c13;
    register __m256d c20,c21,c22,c23;
    //totally 3*4*4 = 48 8float/per refresh
    //__m256d zero = _mm256_set1_pd(0.0);
    c00 = _mm256_loadu_pd(&C[0*lda + 0]);
    c01 = _mm256_loadu_pd(&C[0*lda + 4]);
    c02 = _mm256_loadu_pd(&C[0*lda + 8]);
    c03 = _mm256_loadu_pd(&C[0*lda + 12]);
    c10 = _mm256_loadu_pd(&C[1*lda + 0]);
    c11 = _mm256_loadu_pd(&C[1*lda + 4]);
    c12 = _mm256_loadu_pd(&C[1*lda + 8]);
    c13 = _mm256_loadu_pd(&C[1*lda + 12]);
    c20 = _mm256_loadu_pd(&C[2*lda + 0]);
    c21 = _mm256_loadu_pd(&C[2*lda + 4]);
    c22 = _mm256_loadu_pd(&C[2*lda + 8]);
    c23 = _mm256_loadu_pd(&C[2*lda + 12]);
    for(int p = 0; p < K; p++){
        // register __m256d b1 = _mm256_loadu_pd(&B[p*lda + bi*8]);//4 8float
        // register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+bi*8]);//4 8float
        //if(ai == 0 && bi == 0){
        register __m256d a1 = _mm256_broadcast_sd(&A[0*lda+p]);
        //register __m256d a2 = _mm256_broadcast_sd(&A[0*lda+p]);
        // register __m256d a2 = a1;
        register __m256d b1 = _mm256_loadu_pd(&B[p*lda + 0*8]);//4 8float
        register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]);//4 8float
        c00 = _mm256_fmadd_pd(a1,b1,c00);
        c01 = _mm256_fmadd_pd(a1,b2,c01);

        //else if(ai == 0 && bi == 1){
        // a1 = _mm256_broadcast_sd(&A[0*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[0*lda+p]);
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
        c02 = _mm256_fmadd_pd(a1,b1,c02);
        c03 = _mm256_fmadd_pd(a1,b2,c03);

        //else if(ai == 1 && bi == 0){
        b1 = _mm256_loadu_pd(&B[p*lda + 0*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]);//4 8float
        a1 = _mm256_broadcast_sd(&A[1*lda+p]);
        // a2 = a1;
        // a2 = _mm256_broadcast_sd(&A[1*lda+p]);
        c10 = _mm256_fmadd_pd(a1,b1,c10);
        c11 = _mm256_fmadd_pd(a1,b2,c11);


        //else if(ai == 1 && bi == 1){
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
        // a1 = _mm256_broadcast_sd(&A[1*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[1*lda+p]);
        c12 = _mm256_fmadd_pd(a1,b1,c12);
        c13 = _mm256_fmadd_pd(a1,b2,c13);

        //else if(ai == 2 && bi == 0){
        b1 = _mm256_loadu_pd(&B[p*lda + 0*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]);//4 8float
        a1 = _mm256_broadcast_sd(&A[2*lda+p]);
        // a2 = a1;
        // a2 = _mm256_broadcast_sd(&A[2*lda+p]);
        c20 = _mm256_fmadd_pd(a1,b1,c20);
        c21 = _mm256_fmadd_pd(a1,b2,c21);


        //else if(ai == 2 && bi == 1){
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
        // a1 = _mm256_broadcast_sd(&A[2*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[2*lda+p]);
        c22 = _mm256_fmadd_pd(a1,b1,c22);
        c23 = _mm256_fmadd_pd(a1,b2,c23);


        // if no enough A, then less A is needed;
        // so we have to consider unsafe M and N, but K is safe

    }


    _mm256_storeu_pd(&C[0*lda+0*8], c00);
    _mm256_storeu_pd(&C[0*lda+4+0*8], c01);

    _mm256_storeu_pd(&C[0*lda+1*8], c02);
    _mm256_storeu_pd(&C[0*lda+4+1*8], c03);

    _mm256_storeu_pd(&C[1*lda+0*8], c10);
    _mm256_storeu_pd(&C[1*lda+4+0*8], c11);

    _mm256_storeu_pd(&C[1*lda+1*8], c12);
    _mm256_storeu_pd(&C[1*lda+4+1*8], c13);

    _mm256_storeu_pd(&C[2*lda+0*8], c20);
    _mm256_storeu_pd(&C[2*lda+4+0*8], c21);

    _mm256_storeu_pd(&C[2*lda+1*8], c22);
    _mm256_storeu_pd(&C[2*lda+4+1*8], c23);
}




static inline void do_block2_16(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C){
    register __m256d c00,c01,c02,c03;
    register __m256d c10,c11,c12,c13;
    //totally 3*4*4 = 48 8float/per refresh
    //__m256d zero = _mm256_set1_pd(0.0);
    c00 = _mm256_loadu_pd(&C[0*lda + 0]);
    c01 = _mm256_loadu_pd(&C[0*lda + 4]);
    c02 = _mm256_loadu_pd(&C[0*lda + 8]);
    c03 = _mm256_loadu_pd(&C[0*lda + 12]);
    c10 = _mm256_loadu_pd(&C[1*lda + 0]);
    c11 = _mm256_loadu_pd(&C[1*lda + 4]);
    c12 = _mm256_loadu_pd(&C[1*lda + 8]);
    c13 = _mm256_loadu_pd(&C[1*lda + 12]);
    for(int p = 0; p < K; p++){
        // register __m256d b1 = _mm256_loadu_pd(&B[p*lda + bi*8]);//4 8float
        // register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+bi*8]);//4 8float
        //if(ai == 0 && bi == 0){
        register __m256d a1 = _mm256_broadcast_sd(&A[0*lda+p]);
        // register __m256d a2 = _mm256_broadcast_sd(&A[0*lda+p]);
        register __m256d b1 = _mm256_loadu_pd(&B[p*lda + 0*8]);//4 8float
        register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]);//4 8float
        c00 = _mm256_fmadd_pd(a1,b1,c00);
        c01 = _mm256_fmadd_pd(a1,b2,c01);

        //else if(ai == 0 && bi == 1){
        // a1 = _mm256_broadcast_sd(&A[0*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[0*lda+p]);
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
        c02 = _mm256_fmadd_pd(a1,b1,c02);
        c03 = _mm256_fmadd_pd(a1,b2,c03);

        //else if(ai == 1 && bi == 0){
        b1 = _mm256_loadu_pd(&B[p*lda + 0*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]);//4 8float
        a1 = _mm256_broadcast_sd(&A[1*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[1*lda+p]);
        c10 = _mm256_fmadd_pd(a1,b1,c10);
        c11 = _mm256_fmadd_pd(a1,b2,c11);


        //else if(ai == 1 && bi == 1){
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
        // a1 = _mm256_broadcast_sd(&A[1*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[1*lda+p]);
        c12 = _mm256_fmadd_pd(a1,b1,c12);
        c13 = _mm256_fmadd_pd(a1,b2,c13);


        // if no enough A, then less A is needed;
        // so we have to consider unsafe M and N, but K is safe

    }


    _mm256_storeu_pd(&C[0*lda+0*8], c00);
    _mm256_storeu_pd(&C[0*lda+4+0*8], c01);

    _mm256_storeu_pd(&C[0*lda+1*8], c02);
    _mm256_storeu_pd(&C[0*lda+4+1*8], c03);

    _mm256_storeu_pd(&C[1*lda+0*8], c10);
    _mm256_storeu_pd(&C[1*lda+4+0*8], c11);

    _mm256_storeu_pd(&C[1*lda+1*8], c12);
    _mm256_storeu_pd(&C[1*lda+4+1*8], c13);
}

static inline void do_block1_16(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C){
    register __m256d c00,c01,c02,c03;
    //totally 3*4*4 = 48 8float/per refresh
    //__m256d zero = _mm256_set1_pd(0.0);
    c00 = _mm256_loadu_pd(&C[0*lda + 0]);
    c01 = _mm256_loadu_pd(&C[0*lda + 4]);
    c02 = _mm256_loadu_pd(&C[0*lda + 8]);
    c03 = _mm256_loadu_pd(&C[0*lda + 12]);
    for(int p = 0; p < K; p++){
        // register __m256d b1 = _mm256_loadu_pd(&B[p*lda + bi*8]);//4 8float
        // register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+bi*8]);//4 8float
        //if(ai == 0 && bi == 0){
        register __m256d a1 = _mm256_broadcast_sd(&A[0*lda+p]);
        // register __m256d a2 = _mm256_broadcast_sd(&A[0*lda+p]);
        register __m256d b1 = _mm256_loadu_pd(&B[p*lda + 0*8]);//4 8float
        register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]);//4 8float
        c00 = _mm256_fmadd_pd(a1,b1,c00);
        c01 = _mm256_fmadd_pd(a1,b2,c01);

        //else if(ai == 0 && bi == 1){
        // a1 = _mm256_broadcast_sd(&A[0*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[0*lda+p]);
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
        c02 = _mm256_fmadd_pd(a1,b1,c02);
        c03 = _mm256_fmadd_pd(a1,b2,c03);


        // if no enough A, then less A is needed;
        // so we have to consider unsafe M and N, but K is safe

    }


    _mm256_storeu_pd(&C[0*lda+0*8], c00);
    _mm256_storeu_pd(&C[0*lda+4+0*8], c01);

    _mm256_storeu_pd(&C[0*lda+1*8], c02);
    _mm256_storeu_pd(&C[0*lda+4+1*8], c03);
}
