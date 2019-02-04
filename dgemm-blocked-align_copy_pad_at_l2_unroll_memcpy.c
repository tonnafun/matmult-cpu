
#include <immintrin.h>
//#include <x86intrin.h>
#include <avx2intrin.h>
#include <stdint.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";


/**
lv1 cache: 32k
lv2 cache: 256k
lv3 cache: 10k+k
**/


//128 - 500
#define REGA 3
#define REGB 4 // B = 4*4
//#define BLOCK_SIZE1 96
#define BLOCK_SIZE2 192

#define L1_BLOCK_SIZE_M 48
#define L1_BLOCK_SIZE_N 32
#define L1_BLOCK_SIZE_K 32

#define REG_BLOCK_SIZE_M REGA
#define REG_BLOCK_SIZE_N REGB * 4
#define REG_BLOCK_SIZE_K L1_BLOCK_SIZE_K


#define min(a,b) (((a)<(b))?(a):(b))


//M = REGA = 3, N = REGB*256/64 = 16
//for block1, M = 3, N = 16, which means all c00-c13 are stored in C
//K changeable
static inline void avx_kernel(int K, double* restrict A, double* restrict B, double* restrict C) {
    register __m256d c00,c01,c02,c03;
    register __m256d c10,c11,c12,c13;
    register __m256d c20,c21,c22,c23;
    //totally 3*4*4 = 48 8float/per refresh
    //__m256d zero = _mm256_set1_pd(0.0);
    c00 = _mm256_loadu_pd(&C[0 * BLOCK_SIZE2 + 0]);
    c01 = _mm256_loadu_pd(&C[0 * BLOCK_SIZE2 + 4]);
    c02 = _mm256_loadu_pd(&C[0 * BLOCK_SIZE2 + 8]);
    c03 = _mm256_loadu_pd(&C[0 * BLOCK_SIZE2 + 12]);
    c10 = _mm256_loadu_pd(&C[1 * BLOCK_SIZE2 + 0]);
    c11 = _mm256_loadu_pd(&C[1 * BLOCK_SIZE2 + 4]);
    c12 = _mm256_loadu_pd(&C[1 * BLOCK_SIZE2 + 8]);
    c13 = _mm256_loadu_pd(&C[1 * BLOCK_SIZE2 + 12]);
    c20 = _mm256_loadu_pd(&C[2 * BLOCK_SIZE2 + 0]);
    c21 = _mm256_loadu_pd(&C[2 * BLOCK_SIZE2 + 4]);
    c22 = _mm256_loadu_pd(&C[2 * BLOCK_SIZE2 + 8]);
    c23 = _mm256_loadu_pd(&C[2 * BLOCK_SIZE2 + 12]);
    for(int p = 0; p < K; ++p){
        // register __m256d b1 = _mm256_loadu_pd(&B[p*lda + bi*8]);//4 8float
        // register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+bi*8]);//4 8float
        //if(ai == 0 && bi == 0){
        register __m256d a1 = _mm256_broadcast_sd(&A[0 * BLOCK_SIZE2 + p]);
        //register __m256d a2 = _mm256_broadcast_sd(&A[0*lda+p]);
        // register __m256d a2 = a1;
        register __m256d b1 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 + 0*8]);//4 8float
        register __m256d b2 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 + 4 + 0*8]);//4 8float

        c00 = _mm256_fmadd_pd(a1,b1,c00);
        c01 = _mm256_fmadd_pd(a1,b2,c01);

        //else if(ai == 0 && bi == 1){
        // a1 = _mm256_broadcast_sd(&A[0*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[0*lda+p]);
        b1 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 +4+1*8]);//4 8float
        c02 = _mm256_fmadd_pd(a1,b1,c02);
        c03 = _mm256_fmadd_pd(a1,b2,c03);

        //else if(ai == 1 && bi == 0){
        b1 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 + 0*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 +4+0*8]);//4 8float
        a1 = _mm256_broadcast_sd(&A[1 * BLOCK_SIZE2 +p]);
        // a2 = a1;
        // a2 = _mm256_broadcast_sd(&A[1*lda+p]);
        c10 = _mm256_fmadd_pd(a1,b1,c10);
        c11 = _mm256_fmadd_pd(a1,b2,c11);


        //else if(ai == 1 && bi == 1){
        b1 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 +4+1*8]);//4 8float
        // a1 = _mm256_broadcast_sd(&A[1*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[1*lda+p]);
        c12 = _mm256_fmadd_pd(a1,b1,c12);
        c13 = _mm256_fmadd_pd(a1,b2,c13);

        //else if(ai == 2 && bi == 0){
        b1 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 + 0*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 +4+0*8]);//4 8float
        a1 = _mm256_broadcast_sd(&A[2* BLOCK_SIZE2 +p]);
        // a2 = a1;
        // a2 = _mm256_broadcast_sd(&A[2*lda+p]);
        c20 = _mm256_fmadd_pd(a1,b1,c20);
        c21 = _mm256_fmadd_pd(a1,b2,c21);


        //else if(ai == 2 && bi == 1){
        b1 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p * BLOCK_SIZE2 +4+1*8]);//4 8float
        // a1 = _mm256_broadcast_sd(&A[2*lda+p]);
        // a2 = _mm256_broadcast_sd(&A[2*lda+p]);
        c22 = _mm256_fmadd_pd(a1,b1,c22);
        c23 = _mm256_fmadd_pd(a1,b2,c23);


        // if no enough A, then less A is needed;
        // so we have to consider unsafe M and N, but K is safe

    }


    _mm256_storeu_pd(&C[0 * BLOCK_SIZE2 +0*8], c00);
    _mm256_storeu_pd(&C[0 * BLOCK_SIZE2 +4+0*8], c01);

    _mm256_storeu_pd(&C[0 * BLOCK_SIZE2 +1*8], c02);
    _mm256_storeu_pd(&C[0 * BLOCK_SIZE2 +4+1*8], c03);

    _mm256_storeu_pd(&C[1 * BLOCK_SIZE2 +0*8], c10);
    _mm256_storeu_pd(&C[1 * BLOCK_SIZE2 +4+0*8], c11);

    _mm256_storeu_pd(&C[1 * BLOCK_SIZE2 +1*8], c12);
    _mm256_storeu_pd(&C[1 * BLOCK_SIZE2 +4+1*8], c13);

    _mm256_storeu_pd(&C[2 * BLOCK_SIZE2 +0*8], c20);
    _mm256_storeu_pd(&C[2 * BLOCK_SIZE2 +4+0*8], c21);

    _mm256_storeu_pd(&C[2 * BLOCK_SIZE2 +1*8], c22);
    _mm256_storeu_pd(&C[2 * BLOCK_SIZE2 +4+1*8], c23);
}


static inline void do_block_1(int M, int N, int K, double* restrict A_padded, double* restrict B_padded, double* restrict C_padded) {
    for (int i = 0; i < M; i += REG_BLOCK_SIZE_M)
        for (int j = 0; j < N; j += REG_BLOCK_SIZE_N)
            for (int k = 0; k < K; k += REG_BLOCK_SIZE_K) {
                int curK = min (REG_BLOCK_SIZE_K, K - k);

                avx_kernel(curK,
                           A_padded + i * BLOCK_SIZE2 + k,
                           B_padded + k * BLOCK_SIZE2 + j,
                           C_padded + i * BLOCK_SIZE2 + j);
            }
}


static inline void do_block_2(int M, int N, int K, double* restrict A_padded, double* restrict B_padded, double* restrict C_padded) {

    for (int i = 0; i < M; i += L1_BLOCK_SIZE_M) {
        int curM = min (L1_BLOCK_SIZE_M, M - i);

        for (int j = 0; j < N; j += L1_BLOCK_SIZE_N) {
            int curN = min (L1_BLOCK_SIZE_N, N - j);

            for (int k = 0; k < K; k += L1_BLOCK_SIZE_K) {
                int curK = min (L1_BLOCK_SIZE_K, K - k);

                do_block_1(curM, curN, curK,
                           A_padded + i * BLOCK_SIZE2 + k,
                           B_padded + k * BLOCK_SIZE2 + j,
                           C_padded + i * BLOCK_SIZE2 + j);
            }
        }
    }
}


void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C) {
    double __attribute__(( aligned(__BIGGEST_ALIGNMENT__))) C_padded[BLOCK_SIZE2][BLOCK_SIZE2] = {0};
    double __attribute__(( aligned(__BIGGEST_ALIGNMENT__))) A_padded[BLOCK_SIZE2][BLOCK_SIZE2] = {0};
    double __attribute__(( aligned(__BIGGEST_ALIGNMENT__))) B_padded[BLOCK_SIZE2][BLOCK_SIZE2] = {0};

    for (int i = 0; i < lda; i += BLOCK_SIZE2) {
        int curM = min (BLOCK_SIZE2, lda - i);

        for (int j = 0; j < lda; j += BLOCK_SIZE2) {
            int curN = min (BLOCK_SIZE2, lda - j);

            int i_lda = i * lda;
            int i_lda_plus_j = i_lda + j;

            // ---------------
            int ii = 0;
            int block_limit = (curM / 8) * 8;
            while (ii < block_limit) {
                memcpy(C_padded[ii], C + i_lda_plus_j + ii * lda, sizeof(double) * curN);
                memcpy(C_padded[ii + 1], C + i_lda_plus_j + (ii + 1) * lda, sizeof(double) * curN);
                memcpy(C_padded[ii + 2], C + i_lda_plus_j + (ii + 2) * lda, sizeof(double) * curN);
                memcpy(C_padded[ii + 3], C + i_lda_plus_j + (ii + 3) * lda, sizeof(double) * curN);
                memcpy(C_padded[ii + 4], C + i_lda_plus_j + (ii + 4) * lda, sizeof(double) * curN);
                memcpy(C_padded[ii + 5], C + i_lda_plus_j + (ii + 5) * lda, sizeof(double) * curN);
                memcpy(C_padded[ii + 6], C + i_lda_plus_j + (ii + 6) * lda, sizeof(double) * curN);
                memcpy(C_padded[ii + 7], C + i_lda_plus_j + (ii + 7) * lda, sizeof(double) * curN);
                ii += 8;
            }
            if (ii < curM) {
                switch (curM - ii) {
                    case 7 : memcpy(C_padded[ii], C + i_lda_plus_j + ii * lda, sizeof(double) * curN); ii++;
                    case 6 : memcpy(C_padded[ii], C + i_lda_plus_j + ii * lda, sizeof(double) * curN); ii++;
                    case 5 : memcpy(C_padded[ii], C + i_lda_plus_j + ii * lda, sizeof(double) * curN); ii++;
                    case 4 : memcpy(C_padded[ii], C + i_lda_plus_j + ii * lda, sizeof(double) * curN); ii++;
                    case 3 : memcpy(C_padded[ii], C + i_lda_plus_j + ii * lda, sizeof(double) * curN); ii++;
                    case 2 : memcpy(C_padded[ii], C + i_lda_plus_j + ii * lda, sizeof(double) * curN); ii++;
                    case 1 : memcpy(C_padded[ii], C + i_lda_plus_j + ii * lda, sizeof(double) * curN);
                }
            }
            // ---------------

            for (int k = 0; k < lda; k += BLOCK_SIZE2) {
                int i_lda_plus_k = i_lda + k;
                int k_lda_plus_j = k * lda + j;

                int curK = min (BLOCK_SIZE2, lda - k);

                // ---------------
                int ii = 0;
                int block_limit = (curM / 8) * 8;
                while (ii < block_limit) {
                    memcpy(A_padded[ii], A + i_lda_plus_k + ii * lda, sizeof(double) * curK);
                    memcpy(A_padded[ii + 1], A + i_lda_plus_k + (ii + 1) * lda, sizeof(double) * curK);
                    memcpy(A_padded[ii + 2], A + i_lda_plus_k + (ii + 2) * lda, sizeof(double) * curK);
                    memcpy(A_padded[ii + 3], A + i_lda_plus_k + (ii + 3) * lda, sizeof(double) * curK);
                    memcpy(A_padded[ii + 4], A + i_lda_plus_k + (ii + 4) * lda, sizeof(double) * curK);
                    memcpy(A_padded[ii + 5], A + i_lda_plus_k + (ii + 5) * lda, sizeof(double) * curK);
                    memcpy(A_padded[ii + 6], A + i_lda_plus_k + (ii + 6) * lda, sizeof(double) * curK);
                    memcpy(A_padded[ii + 7], A + i_lda_plus_k + (ii + 7) * lda, sizeof(double) * curK);
                    ii += 8;
                }
                if (ii < curM) {
                    switch (curM - ii) {
                        case 7 : memcpy(A_padded[ii], A + i_lda_plus_k + ii * lda, sizeof(double) * curK); ii++;
                        case 6 : memcpy(A_padded[ii], A + i_lda_plus_k + ii * lda, sizeof(double) * curK); ii++;
                        case 5 : memcpy(A_padded[ii], A + i_lda_plus_k + ii * lda, sizeof(double) * curK); ii++;
                        case 4 : memcpy(A_padded[ii], A + i_lda_plus_k + ii * lda, sizeof(double) * curK); ii++;
                        case 3 : memcpy(A_padded[ii], A + i_lda_plus_k + ii * lda, sizeof(double) * curK); ii++;
                        case 2 : memcpy(A_padded[ii], A + i_lda_plus_k + ii * lda, sizeof(double) * curK); ii++;
                        case 1 : memcpy(A_padded[ii], A + i_lda_plus_k + ii * lda, sizeof(double) * curK);
                    }
                }
                // ---------------

                // ---------------
                int kk = 0;
                block_limit = (curK / 8) * 8;
                while (kk < block_limit) {
                    memcpy(B_padded[kk], B + k_lda_plus_j + kk * lda, sizeof(double) * curN);
                    memcpy(B_padded[kk + 1], B + k_lda_plus_j + (kk + 1) * lda, sizeof(double) * curN);
                    memcpy(B_padded[kk + 2], B + k_lda_plus_j + (kk + 2) * lda, sizeof(double) * curN);
                    memcpy(B_padded[kk + 3], B + k_lda_plus_j + (kk + 3) * lda, sizeof(double) * curN);
                    memcpy(B_padded[kk + 4], B + k_lda_plus_j + (kk + 4) * lda, sizeof(double) * curN);
                    memcpy(B_padded[kk + 5], B + k_lda_plus_j + (kk + 5) * lda, sizeof(double) * curN);
                    memcpy(B_padded[kk + 6], B + k_lda_plus_j + (kk + 6) * lda, sizeof(double) * curN);
                    memcpy(B_padded[kk + 7], B + k_lda_plus_j + (kk + 7) * lda, sizeof(double) * curN);
                    kk += 8;
                }
                if (kk < curK) {
                    switch (curK - kk) {
                        case 7 : memcpy(B_padded[kk], B + k_lda_plus_j + kk * lda, sizeof(double) * curN); kk++;
                        case 6 : memcpy(B_padded[kk], B + k_lda_plus_j + kk * lda, sizeof(double) * curN); kk++;
                        case 5 : memcpy(B_padded[kk], B + k_lda_plus_j + kk * lda, sizeof(double) * curN); kk++;
                        case 4 : memcpy(B_padded[kk], B + k_lda_plus_j + kk * lda, sizeof(double) * curN); kk++;
                        case 3 : memcpy(B_padded[kk], B + k_lda_plus_j + kk * lda, sizeof(double) * curN); kk++;
                        case 2 : memcpy(B_padded[kk], B + k_lda_plus_j + kk * lda, sizeof(double) * curN); kk++;
                        case 1 : memcpy(B_padded[kk], B + k_lda_plus_j + kk * lda, sizeof(double) * curN); kk++;
                    }
                }
                // ---------------

                do_block_2(curM, curN, curK, A_padded, B_padded, C_padded);

            }

            // ---------------
            ii = 0;
            block_limit = (curM / 8) * 8;
            while (ii < block_limit) {
                memcpy(C + i_lda_plus_j + ii * lda, C_padded[ii], sizeof(double) * curN);
                memcpy(C + i_lda_plus_j + (ii + 1) * lda, C_padded[ii + 1], sizeof(double) * curN);
                memcpy(C + i_lda_plus_j + (ii + 2) * lda, C_padded[ii + 2], sizeof(double) * curN);
                memcpy(C + i_lda_plus_j + (ii + 3) * lda, C_padded[ii + 3], sizeof(double) * curN);
                memcpy(C + i_lda_plus_j + (ii + 4) * lda, C_padded[ii + 4], sizeof(double) * curN);
                memcpy(C + i_lda_plus_j + (ii + 5) * lda, C_padded[ii + 5], sizeof(double) * curN);
                memcpy(C + i_lda_plus_j + (ii + 6) * lda, C_padded[ii + 6], sizeof(double) * curN);
                memcpy(C + i_lda_plus_j + (ii + 7) * lda, C_padded[ii + 7], sizeof(double) * curN);
                ii += 8;
            }
            if (ii < curM) {
                switch (curM - ii) {
                    case 7 : memcpy(C + i_lda_plus_j + ii * lda, C_padded[ii], sizeof(double) * curN); ii++;
                    case 6 : memcpy(C + i_lda_plus_j + ii * lda, C_padded[ii], sizeof(double) * curN); ii++;
                    case 5 : memcpy(C + i_lda_plus_j + ii * lda, C_padded[ii], sizeof(double) * curN); ii++;
                    case 4 : memcpy(C + i_lda_plus_j + ii * lda, C_padded[ii], sizeof(double) * curN); ii++;
                    case 3 : memcpy(C + i_lda_plus_j + ii * lda, C_padded[ii], sizeof(double) * curN); ii++;
                    case 2 : memcpy(C + i_lda_plus_j + ii * lda, C_padded[ii], sizeof(double) * curN); ii++;
                    case 1 : memcpy(C + i_lda_plus_j + ii * lda, C_padded[ii], sizeof(double) * curN);
                }
            }
            // ---------------
        }
    }
}
