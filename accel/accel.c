// accel.c — BLAS-accelerated kernels for molequla inference (double precision)
//
// macOS: Apple Accelerate (built-in, zero deps)
// Linux: OpenBLAS (apt install libopenblas-dev)

#include "accel.h"

#ifdef __APPLE__
  #ifndef ACCELERATE_NEW_LAPACK
    #define ACCELERATE_NEW_LAPACK
  #endif
  #include <Accelerate/Accelerate.h>
#else
  #include <cblas.h>
#endif

// y = A @ x  (double precision matrix-vector)
void accel_dgemv(int M, int N, const double *A, const double *x, double *y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N,
                1.0, A, N, x, 1, 0.0, y, 1);
}

// C = alpha * A @ B + beta * C  (double precision matrix-matrix)
void accel_dgemm(int M, int N, int K,
                 double alpha, const double *A, const double *B,
                 double beta, double *C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, K, B, N, beta, C, N);
}

// dot product (double precision)
double accel_ddot(int N, const double *x, const double *y) {
    return cblas_ddot(N, x, 1, y, 1);
}
