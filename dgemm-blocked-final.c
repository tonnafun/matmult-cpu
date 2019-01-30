
#include <immintrin.h>
//#include <x86intrin.h>
#include <avx2intrin.h>
#include <stdint.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#define REG_A 3
#define REG_B 4  // B is a 4*4 matrix
#define BLOCK_SIZE1 96
#define BLOCK_SIZE2 96

#define min(a,b) (((a)<(b))?(a):(b))


static inline void do_block_naive(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            double r = A[i * lda + k];
            for (int j = 0; j < N; ++j) {
                C[i * lda + j] += r * B[k * lda + j];
            }
        }
}


// C[3x16] = A[3xBLOCK_SIZE1] x B[BLOCK_SIZE1x16]
static inline void do_block_3x16(int lda, int M, int N, int K, double * restrict A, double * restrict B, double * restrict C) {
    register __m256d c00,c01,c02,c03;
    register __m256d c10,c11,c12,c13;
    register __m256d c20,c21,c22,c23;

    c00 = _mm256_loadu_pd(C + 0*lda + 0);
    c01 = _mm256_loadu_pd(C + 0*lda + 4);
    c02 = _mm256_loadu_pd(C + 0*lda + 8);
    c03 = _mm256_loadu_pd(C + 0*lda + 12);
    c10 = _mm256_loadu_pd(C + 1*lda + 0);
    c11 = _mm256_loadu_pd(C + 1*lda + 4);
    c12 = _mm256_loadu_pd(C + 1*lda + 8);
    c13 = _mm256_loadu_pd(C + 1*lda + 12);
    c20 = _mm256_loadu_pd(C + 2*lda + 0);
    c21 = _mm256_loadu_pd(C + 2*lda + 4);
    c22 = _mm256_loadu_pd(C + 2*lda + 8);
    c23 = _mm256_loadu_pd(C + 2*lda + 12);

    for(int p = 0; p < K; p++) {
        // register __m256d b1 = _mm256_loadu_pd(&B[p*lda + bi*8]);//4 8float
        // register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+bi*8]);//4 8float
        //if(ai == 0 && bi == 0){
//        register __m256d a1 = _mm256_broadcast_sd(A + 0*lda + p);
        register __m256d a = _mm256_broadcast_sd(A + 0*lda + p);
        register __m256d b1 = _mm256_loadu_pd(&B[p*lda + 0*8]); //4 8float
        register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]); //4 8float
        c00 = _mm256_fmadd_pd(a, b1, c00);
        c01 = _mm256_fmadd_pd(a, b2, c01);

        //else if(ai == 0 && bi == 1){
//        a1 = _mm256_broadcast_sd(A + 0*lda + p);
        a = _mm256_broadcast_sd(A + 0*lda + p);
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
        c02 = _mm256_fmadd_pd(a,b1,c02);
        c03 = _mm256_fmadd_pd(a,b2,c03);

        //else if(ai == 1 && bi == 0){
        b1 = _mm256_loadu_pd(&B[p*lda + 0*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]);//4 8float
        a = _mm256_broadcast_sd(&A[1*lda+p]);
//        a2 = _mm256_broadcast_sd(&A[1*lda+p]);
        c10 = _mm256_fmadd_pd(a,b1,c10);
        c11 = _mm256_fmadd_pd(a,b2,c11);


        //else if(ai == 1 && bi == 1){
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
        a = _mm256_broadcast_sd(&A[1*lda+p]);
//        a2 = _mm256_broadcast_sd(&A[1*lda+p]);
        c12 = _mm256_fmadd_pd(a,b1,c12);
        c13 = _mm256_fmadd_pd(a,b2,c13);

        //else if(ai == 2 && bi == 0){
        b1 = _mm256_loadu_pd(&B[p*lda + 0*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+0*8]);//4 8float
        a = _mm256_broadcast_sd(&A[2*lda+p]);
//        a2 = _mm256_broadcast_sd(&A[2*lda+p]);
        c20 = _mm256_fmadd_pd(a,b1,c20);
        c21 = _mm256_fmadd_pd(a,b2,c21);


        //else if(ai == 2 && bi == 1){
        b1 = _mm256_loadu_pd(&B[p*lda + 1*8]);//4 8float
        b2 = _mm256_loadu_pd(&B[p*lda+4+1*8]);//4 8float
//        a1 = _mm256_broadcast_sd(&A[2*lda+p]);
        a = _mm256_broadcast_sd(&A[2*lda+p]);
        c22 = _mm256_fmadd_pd(a,b1,c22);
        c23 = _mm256_fmadd_pd(a,b2,c23);


        // if no enough A, then less A is needed;
        // so we have to consider unsafe M and N, but K is safe

    }


    _mm256_storeu_pd(C + 0*lda + 0*8, c00);
    _mm256_storeu_pd(C + 0*lda + 4 + 0*8, c01);

    _mm256_storeu_pd(C + 0*lda + 1*8, c02);
    _mm256_storeu_pd(C + 0*lda + 4 + 1*8, c03);

    _mm256_storeu_pd(C + 1*lda + 0*8, c10);
    _mm256_storeu_pd(C + 1*lda + 4 + 0*8, c11);

    _mm256_storeu_pd(C + 1*lda + 1*8, c12);
    _mm256_storeu_pd(C + 1*lda + 4 + 1*8, c13);

    _mm256_storeu_pd(C + 2*lda + 0*8, c20);
    _mm256_storeu_pd(C + 2*lda + 4 + 0*8, c21);

    _mm256_storeu_pd(C + 2*lda + 1*8, c22);
    _mm256_storeu_pd(C + 2*lda + 4 + 1*8, c23);
}






static inline void do_block1(int lda, int M, int N, int K, double * restrict A, double * restrict B, double * restrict C) {
    for (int i = 0; i < M; i += REG_A)
        for (int j = 0; j < N; j += REG_B * 4)
            for (int k = 0; k < K; k += BLOCK_SIZE1) {
                int curM = min (REG_A, M - i);
                int curN = min (REG_B * 4, N - j);
                int curK = min (BLOCK_SIZE1, K - k);

                // A: curM x curK
                // B: curK x curN
                // C: curM x curN
                if (curM == REG_A && curN == REG_B * 4)
                    do_block_3x16(lda, curM, curN, curK,
                                  A + i * lda + k,
                                  B + k * lda + j,
                                  C + i * lda + j
                    );
                else
                    do_block_naive(lda, curM, curN, curK,
                                  A + i * lda + k,
                                  B + k * lda + j,
                                  C + i * lda + j
                    );

            }
}

static inline void do_block2(int lda, int M, int N, int K, double * restrict A, double * restrict B, double * restrict C) {
    for (int i = 0; i < M; i += BLOCK_SIZE2)
        for (int j = 0; j < N; j += BLOCK_SIZE2)
            for (int k = 0; k < K; k += BLOCK_SIZE2) {
                int curM = min (BLOCK_SIZE2, M - i);
                int curN = min (BLOCK_SIZE2, N - j);
                int curK = min (BLOCK_SIZE2, K - k);

                do_block1(lda, curM, curN, curK,
                          A + i * lda + k,
                          B + k * lda + j,
                          C + i * lda + j
                );
            }
}

void square_dgemm(int lda, double * restrict A, double * restrict B, double * restrict C) {
    do_block1(lda, lda, lda, lda, A, B, C);
}