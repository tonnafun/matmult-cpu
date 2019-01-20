
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE3)
#define BLOCK_SIZE3 128
#endif

#if !defined(BLOCK_SIZE2)
#define BLOCK_SIZE2 64
#endif

#if !defined(BLOCK_SIZE1)
#define BLOCK_SIZE1 24
#endif

#define min(a,b) (((a)<(b))?(a):(b))


//#define TRANSPOSE

//inline static void do_block1(int lda, int M, int N, int K, double* A, double* B, double* C) {
//  for (int i = 0; i < M; ++i)
//    for (int j = 0; j < N; ++j) {
//      double cij = C[i*lda + j];
//      for (int k = 0; k < K; ++k) {
//#ifdef TRANSPOSE
//        cij += A[i * lda + k] * B[j * lda + k];
//#else
//        cij += A[i * lda + k] * B[k * lda + j];
//#endif
//        C[i * lda + j] = cij;
//      }
//    }
//}

inline static void do_block1(int lda, int M, int N, int K, double* A, double* B, double* C) {
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            double r = A[i * lda + k];
//            double cij = C[i*lda + j];
            for (int j = 0; j < N; ++j) {
#ifdef TRANSPOSE
                C[i * lda + j] += r * B[j * lda + k];
#else
                C[i * lda + j] += r * B[k * lda + j];
#endif
//                C[i * lda + j] = cij;
            }
        }
}

inline static void do_block2 (int lda, int M, int N, int K, double* A, double* B, double* C) {
  for (int i = 0; i < M; i += BLOCK_SIZE1)
    for (int k = 0; k < K; k += BLOCK_SIZE1)
      for (int j = 0; j < N; j += BLOCK_SIZE1) {
        int M1 = min (BLOCK_SIZE1, M - i);
        int N1 = min (BLOCK_SIZE1, N - j);
        int K1 = min (BLOCK_SIZE1, K - k);

#ifdef TRANSPOSE
        do_block1(lda, M1, N1, K1, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
        do_block1(lda, M1, N1, K1, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
      }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
inline static void do_block3 (int lda, int M, int N, int K, double* A, double* B, double* C) {
  /* For each row i of A */
  for (int i = 0; i < M; i += BLOCK_SIZE2)
    /* For each column j of B */
    for (int k = 0; k < K; k += BLOCK_SIZE2)
      for (int j = 0; j < N; j += BLOCK_SIZE2) {
        int M2 = min (BLOCK_SIZE2, M - i);
        int N2 = min (BLOCK_SIZE2, N - j);
        int K2 = min (BLOCK_SIZE2, K - k);

#ifdef TRANSPOSE
        do_block2(lda, M2, N2, K2, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
        do_block2(lda, M2, N2, K2, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
      }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm (int lda, double* A, double* B, double* C)
{
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE3)
    /* For each block-column of B */
    for (int k = 0; k < lda; k += BLOCK_SIZE3)
      /* Accumulate block dgemms into block of C */
      for (int j = 0; j < lda; j += BLOCK_SIZE3)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE3, lda-i);
        int N = min (BLOCK_SIZE3, lda-j);
        int K = min (BLOCK_SIZE3, lda-k);

        /* Perform individual block dgemm */
#ifdef TRANSPOSE
        do_block3(lda, M, N, K, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
        do_block3(lda, M, N, K, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
      }
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
  }
#endif
}

