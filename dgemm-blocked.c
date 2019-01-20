/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <immintrin.h>
//#include <x86intrin.h>
#include <avx2intrin.h>
#include <stdint.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";


/**
lv1 cache: 32k
lv2 cache: 256k
lv3 cache: 6144k
**/


#if !defined(BLOCK_SIZE1)
#define REGA 3
#define REGB 4 // B = 4*4
#define BLOCK_SIZE1 30
#define BLOCK_SIZE BLOCK_SIZE1
#define BLOCK_SIZE2 100
#define BLOCK_SIZE3 104
#endif

#define min(a,b) (((a)<(b))?(a):(b))


// Print the content of a __m256d variable.
void print256(__m256d var) {
    double *val = (double*) &var;
    printf("printf var: %lf %lf %lf %lf\n", 
           val[0], val[1], val[2], val[3]);
}


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
	cij += A[i*lda+k] * B[j*lda+k];
#else
	cij += A[i*lda+k] * B[k*lda+j];
#endif
      C[i*lda+j] = cij;
    }
}


static void do_block2 (int lda, int K, double* restrict A, double* restrict B, double* restrict C)
{
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
  for (int kk = 0; kk < K; kk++){
    register __m256d a0x = _mm256_broadcast_sd(A + kk + 0*lda);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + 1*lda);
    register __m256d b = _mm256_loadu_pd(B + kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd (a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd (a1x, b, c10_c11_c12_c13);
  }
  _mm256_storeu_pd(C, c00_c01_c02_c03);
  _mm256_storeu_pd(C+lda, c10_c11_c12_c13);
}


//M = REGA = 3, N = REGB*256/64 = 16
//for block1, M = 3, N = 16, which means all c00-c13 are stored in C
//K changeable
static void do_block1(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {
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
		for(int bi = 0; bi < REGB/2; bi++){ 
      register __m256d b1 = _mm256_loadu_pd(&B[p*lda + bi*8]);//4 8float
      register __m256d b2 = _mm256_loadu_pd(&B[p*lda+4+bi*8]);//4 8float
      // if no enough B, not considered yet
      for (int ai = 0; ai < M; ai++){
        register __m256d a1 = _mm256_broadcast_sd(&A[ai*lda+p]);
        register __m256d a2 = _mm256_broadcast_sd(&A[ai*lda+p]);
        // a == 0
        if(ai == 0 && bi == 0){
          c00 = _mm256_fmadd_pd(a1,b1,c00);
          c01 = _mm256_fmadd_pd(a2,b2,c01);
        }
        else if(ai == 0 && bi == 1){
          c02 = _mm256_fmadd_pd(a1,b1,c02);
          c03 = _mm256_fmadd_pd(a2,b2,c03);
        }
        // a == 1
        else if(ai == 1 && bi == 0){
          c10 = _mm256_fmadd_pd(a1,b1,c10);
          c11 = _mm256_fmadd_pd(a2,b2,c11);
        }
        else if(ai == 1 && bi == 1){
          c12 = _mm256_fmadd_pd(a1,b1,c12);
          c13 = _mm256_fmadd_pd(a2,b2,c13);
        }
        // a == 2
        else if(ai == 2 && bi == 0){
          c20 = _mm256_fmadd_pd(a1,b1,c20);
          c21 = _mm256_fmadd_pd(a2,b2,c21);
        }
        else if(ai == 2 && bi == 1){
          c22 = _mm256_fmadd_pd(a1,b1,c22);
          c23 = _mm256_fmadd_pd(a2,b2,c23);
        }
      // if no enough A, then less A is needed;
      // so we have to consider unsafe M and N, but K is safe
      }
        //c_ai_bi += _mm256_mul_pd(a1, b1);
        //c_ai_b(i+1) += _mm256_mul_pd(a2, b2);
		}
    }
    for(int ai = 0; ai < M; ai++)
      for(int bi = 0; bi < REGB/2; bi++){
        if(ai == 0 && bi == 0){
          _mm256_storeu_pd(&C[ai*lda+bi*8], c00);
          _mm256_storeu_pd(&C[ai*lda+4+bi*8], c01);
        }
        else if(ai == 0 && bi == 1){
          _mm256_storeu_pd(&C[ai*lda+bi*8], c02);
          _mm256_storeu_pd(&C[ai*lda+4+bi*8], c03);
        }
        else if(ai == 1 && bi == 0){
          _mm256_storeu_pd(&C[ai*lda+bi*8], c10);
          _mm256_storeu_pd(&C[ai*lda+4+bi*8], c11);
        }
        else if(ai == 1 && bi == 1){
          _mm256_storeu_pd(&C[ai*lda+bi*8], c12);
          _mm256_storeu_pd(&C[ai*lda+4+bi*8], c13);
        }
        else if(ai == 2 && bi == 0){
          _mm256_storeu_pd(&C[ai*lda+bi*8], c20);
          _mm256_storeu_pd(&C[ai*lda+4+bi*8], c21);
        }
        else if(ai == 2 && bi == 1){
          _mm256_storeu_pd(&C[ai*lda+bi*8], c22);
          _mm256_storeu_pd(&C[ai*lda+4+bi*8], c23);
        }
      }
	}

static inline void block_square_multilv1(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C){
  for (int i = 0; i < M; i += 2)
    for(int j = 0; j < N; j += 4)
      for(int k = 0; k < K; k += BLOCK_SIZE1){
        int curM = min (2, M-i);
        int curN = min (4, N-j);
        int curK = min (BLOCK_SIZE1, K-k);
#ifdef TRANSPOSE
        if(curM == 2 && curN == 4)
          do_block2(lda, curK, A + i*lda + k, B + j*lda + k, C + i*lda + j);
        else
          do_block(lda, curM, curN, curK, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
        if(curM == 2 && curN == 4)
          do_block2(lda, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);
        else
          do_block(lda, curM, curN, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);
#endif
      }
}

static inline void block_square_multilv2(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C){
  for (int i = 0; i < M; i += BLOCK_SIZE2)
    for(int j = 0; j < N; j += BLOCK_SIZE2)
      for(int k = 0; k < K; k += BLOCK_SIZE2){
        int curM = min (BLOCK_SIZE2, M-i);
        int curN = min (BLOCK_SIZE2, N-j);
        int curK = min (BLOCK_SIZE2, K-k);
#ifdef TRANSPOSE
        block_square_multilv1(lda, curM, curN, curK, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
        block_square_multilv1(lda, curM, curN, curK, A + i*lda + k, B + k*lda + j, C + i*lda + j);
#endif
      }
}



static inline void block_square_multilv3(int lda, double* restrict A, double* restrict B, double* restrict C) {
  for (int i = 0; i < lda; i += BLOCK_SIZE3)
    for(int j = 0; j < lda; j += BLOCK_SIZE3)
      for(int k = 0; k < lda; k += BLOCK_SIZE3) {
        int M = min (BLOCK_SIZE3, lda-i);
        int N = min (BLOCK_SIZE3, lda-j);
        int K = min (BLOCK_SIZE3, lda-k);
#ifdef TRANSPOSE
        block_square_multilv2(lda, M, N, K, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
        block_square_multilv2(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
#endif
      }
}


void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C) {
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
      double t = B[i*lda+j];
      B[i*lda+j] = B[j*lda+i];
      B[j*lda+i] = t;
    }
#endif

  block_square_multilv3(lda, A, B, C);

#if TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
      double t = B[i*lda+j];
      B[i*lda+j] = B[j*lda+i];
      B[j*lda+i] = t;
    }
#endif
}
