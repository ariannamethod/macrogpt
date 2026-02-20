//go:build !(cgo && (darwin || linux))

package accel

func Dgemv(M, N int, A, x, y []float64) {}
func Dgemm(M, N, K int, alpha float64, A, B []float64, beta float64, C_ []float64) {}
func Ddot(N int, x, y []float64) float64 { return 0 }

const HasAccel = false
