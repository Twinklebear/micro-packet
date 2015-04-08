#ifndef MAT3_H
#define MAT3_H

#include "vec.h"

/*
 * Structure storing 8 3x3 matrices
 */
struct Mat3f_8 {
	__m256 mat[3][3];

	// Construct the identity matrix
	inline Mat3f_8(){
		const auto zero = _mm256_set1_ps(0.f);
		const auto one = _mm256_set1_ps(1.f);
		for (int i = 0; i < 3; ++i){
			for (int j = 0; j < 3; ++j){
				if (i == j){
					mat[i][j] = one;
				}
				else {
					mat[i][j] = zero;
				}
			}
		}
	}
	// Construct the matrix from 3 vectors, specifying the values for each row
	inline Mat3f_8(const Vec3f_8 &a, const Vec3f_8 &b, const Vec3f_8 &c){
		for (int i = 0; i < 3; ++i){
			mat[0][i] = a[i];
			mat[1][i] = b[i];
			mat[2][i] = c[i];
		}
	}
	const __m256* operator[](int i) const {
		assert(i < 3);
		return mat[i];
	}
};
// Multiply 8 vectors with the 8 matrices
inline Vec3f_8 operator*(const Mat3f_8 &m, const Vec3f_8 &v){
	Vec3f_8 out;
	for (int i = 0; i < 3; ++i){
		const auto a = _mm256_mul_ps(m[i][0], v.x);
		const auto b = _mm256_fmadd_ps(m[i][1], v.y, a);
		out[i] = _mm256_fmadd_ps(m[i][2], v.z, b);
	}
	return out;
}

#endif

