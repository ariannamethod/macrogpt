//go:build darwin && cgo

package accel

// #cgo CFLAGS: -DACCELERATE_NEW_LAPACK
// #cgo LDFLAGS: -framework Accelerate
// #include "accel.h"
import "C"
import "unsafe"

func Dgemv(M, N int, A, x, y []float64) {
	C.accel_dgemv(C.int(M), C.int(N),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&y[0])))
}

func Dgemm(M, N, K int, alpha float64, A, B []float64, beta float64, C_ []float64) {
	C.accel_dgemm(C.int(M), C.int(N), C.int(K),
		C.double(alpha),
		(*C.double)(unsafe.Pointer(&A[0])),
		(*C.double)(unsafe.Pointer(&B[0])),
		C.double(beta),
		(*C.double)(unsafe.Pointer(&C_[0])))
}

func Ddot(N int, x, y []float64) float64 {
	return float64(C.accel_ddot(C.int(N),
		(*C.double)(unsafe.Pointer(&x[0])),
		(*C.double)(unsafe.Pointer(&y[0]))))
}

const HasAccel = true
