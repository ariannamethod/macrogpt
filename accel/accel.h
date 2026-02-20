#ifndef ACCEL_H
#define ACCEL_H

// BLAS double-precision matrix-vector: y = A @ x
// A: [M x N] row-major, x: [N], y: [M]
void accel_dgemv(int M, int N, const double *A, const double *x, double *y);

// BLAS double-precision matrix-matrix: C = alpha * A @ B + beta * C
// A: [M x K], B: [K x N], C: [M x N] row-major
void accel_dgemm(int M, int N, int K,
                 double alpha, const double *A, const double *B,
                 double beta, double *C);

// Double-precision dot product: sum(x[i] * y[i])
double accel_ddot(int N, const double *x, const double *y);

#endif
