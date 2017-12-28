/*
 * =====================================================================================
 *
 *       Filename:  matrix.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/24 12时04分37秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "matrix0.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>

matrix::matrix(int row, int col) : row(row), col(col)
{
	val = (float*) malloc(sizeof(float) * row * col);
	memset(val, 0, sizeof(float) * row * col);
}

matrix::matrix(const float* val, int row, int col) : row(row), col(col)
{
	this->val = (float*) malloc(sizeof(float) * row * col);
	memcpy(this->val, val, sizeof(float) * row * col);
}

matrix::matrix(const tensor** tensors, int row, int col) : row(row), col(col)
{
	this->val = (float*) malloc(sizeof(float) * row * col);
	for (int i = 0; i < row; i++) {
		memcpy(this->val + i * col, tensors[i]->get_val(), sizeof(float) * col);
	}
}

matrix::matrix(const matrix& other) : row(other.row), col(other.col)
{
	const float* that_val = other.val;
	this->val = (float*) malloc(sizeof(float) * row * col);
	memcpy(this->val, that_val, sizeof(float) * row * col);
}

matrix::~matrix()
{
	if (val) free(val);
}

matrix& matrix::transpose()
{
	float* tmp = val;
	val = (float*) malloc(sizeof(float) * col * row);
	for (int i = 0; i < row; ++i)
		for (int j = 0; j < col; ++j) 
			val[j * row + i] = tmp[i * col + j];

	int r = row;
	row = col;
	col = r;
	free(tmp);

	return *this;
}

matrix* matrix::transpose(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.col, mat.row);
	else {
		assert((*ret)->row * (*ret)->col == mat.col * mat.row);
		(*ret)->row = mat.col;
		(*ret)->col = mat.row;
	}

	matrix* pmat = *ret;
	for (int i = 0; i < pmat->row; ++i)
		for (int j = 0; j < pmat->col; ++j) 
			pmat->val[i * pmat->col + j] = mat.val[j * mat.col + i];

	return *ret;
}

matrix& matrix::plus(const matrix& mat)
{
	const float* that_val = mat.val;
	for (int i = 0; i < row * col; i++)
		val[i] += that_val[i];

	return *this;
}

matrix* matrix::plus(matrix** ret, const matrix& mat0, const matrix& mat1)
{
	if (0 == *ret)
		*ret = new matrix(mat0.row, mat0.col);
	else {
		assert((*ret)->row * (*ret)->col == mat0.row * mat0.col);
		(*ret)->row = mat0.row;
		(*ret)->col = mat0.col;
	}

	matrix* pmat = *ret;

	float* val = pmat->val;
	const float* val0 = mat0.val;
	const float* val1 = mat1.val;
	for (int i = 0; i < pmat->row * pmat->col; ++i)
		val[i] = val0[i] + val1[i];

	return *ret;
}

matrix& matrix::plus_row_row(const matrix& row_matrix)
{
	assert(col == row_matrix.col && 1 == row_matrix.row);
	int j = 0;
	for (int i = 0; i < row * col; i++, j++) {
		if (j >= col) j = 0;
		val[i] += row_matrix.val[j];
	}

	return *this;
}

matrix& matrix::plus_col_col(const matrix& col_matrix)
{
	assert(row == col_matrix.row && 1 == col_matrix.col);
	int c = 0;
	int r = 0;
	for (int i = 0; i < row * col; i++, c++) {
		if (c >= col) {
			c = 0;
			++r;
		}
		val[i] += col_matrix.val[r];
	}

	return *this;
}

matrix& matrix::minus(const matrix& mat)
{
	const float* that_val = mat.val;
	for (int i = 0; i < row * col; i++)
		val[i] -= that_val[i];

	return *this;
}

matrix* matrix::minus(matrix** ret, const matrix& mat0, const matrix& mat1)
{
	if (0 == *ret)
		*ret = new matrix(mat0.row, mat0.col);
	else {
		assert((*ret)->row * (*ret)->col == mat0.row * mat0.col);
		(*ret)->row = mat0.row;
		(*ret)->col = mat0.col;
	}

	matrix* pmat = *ret;

	float* val = pmat->val;
	const float* val0 = mat0.val;
	const float* val1 = mat1.val;
	for (int i = 0; i < pmat->row * pmat->col; ++i)
		val[i] = val0[i] - val1[i];

	return *ret;
}

__global__
matrix* matrix::multiply(matrix** ret, const matrix& mat0, const matrix& mat1)
{
    if (0 == *ret)
        *ret = new matrix(mat0.row, mat1.col);
    else {
        assert((*ret)->row * (*ret)->col == mat0.row * mat1.col);
        (*ret)->row = mat0.row;
        (*ret)->col = mat1.col;
    }

    matrix* pmat = *ret;

    cublasStatus_t status;
    cublasHandle_t handle;
    status = cublasCreate(&handle);

    if (CUBLAS_STATUS_SUCCESS != status) {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "CUBLAS 初始化出错!" << std::endl;
        }
        cublasDestroy(handle);

        // use cpu.
        for (int i = 0; i < pmat->row; ++i) {
            for (int j = 0; j < pmat->col; ++j) {
                float tmp = 0.0f;
                for (int k = 0; k < mat0.col; ++k)
                    tmp += mat0.val[i * mat0.col + k] * mat1.val[k * mat1.col + j];

                pmat->val[i * pmat->col + j] = tmp;
            }
        }
    } else {
        // gpu : cublas.
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, sizeof(float) * mat0.get_row() * mat0.get_col());
        cudaMalloc((void**)&d_B, sizeof(float) * mat1.get_row() * mat1.get_col());
        cudaMalloc((void**)&d_C, sizeof(float) * mat0.get_row() * mat1.get_col());

        cublasSetVector(mat0.get_row() * mat0.get_col(), sizeof(float), mat0.val, 1, d_A, 1);
        cublasSetVector(mat1.get_row() * mat1.get_col(), sizeof(float), mat1.val, 1, d_B, 1);

        cudaThreadSynchronize();

        float a = 1.f;
        float b = 0.f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, mat1.get_col(), mat0.get_row(), mat0.get_col(), &a, d_B, mat1.get_col(), d_A, mat0.get_col(), &b, d_C, mat1.get_col());
        cudaThreadSynchronize();

        cublasGetVector(mat0.get_row() * mat1.get_col(), sizeof(float), d_C, 1, pmat->val, 1);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        cublasDestroy(handle);
    }

    return *ret;
}

matrix& matrix::multiply_num(float coef)
{
	for (int i = 0; i < row * col; ++i)
		val[i] *= coef;
	return *this;
}

matrix& matrix::divide_num(float coef)
{
	for (int i = 0; i < row * col; ++i)
		val[i] /= coef;
	return *this;
}

matrix* matrix::multiply_num(matrix** ret, const matrix& mat, float coef)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	(*ret)->copy_from(mat).multiply_num(coef);

	return *ret;
}

matrix* matrix::divide_num(matrix** ret, const matrix& mat, float coef)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	(*ret)->copy_from(mat).divide_num(coef);

	return *ret;
}

matrix* matrix::sum_rows(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(1, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.col);
		(*ret)->row = 1;
		(*ret)->col = mat.col;
	}

	memset((*ret)->val, 0, sizeof(float) * mat.col);
	for (int i = 0, j = 0; i < mat.row * mat.col; ++i, j++) {
		if (j >= mat.col) j = 0;
		(*ret)->val[j] += mat.val[i];
	}

	return *ret;
}

matrix* matrix::sum_cols(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, 1);
	else {
		assert((*ret)->row * (*ret)->col == mat.row);
		(*ret)->row = mat.row;
		(*ret)->col = 1;
	}

	int index = 0; 
	for (int i = 0; i < mat.row; ++i) {
		(*ret)->val[i] = 0;
		for (int j = 0; j < mat.col; ++j)
			(*ret)->val[i] += mat.val[index++];
	}

	return *ret;
}

matrix& matrix::copy_from(const matrix& other)
{
	assert(row * col == other.row * other.col);
	row = other.row;
	col = other.col;

	memcpy(val, other.val, sizeof(float) * row * col);

	return *this;
}

matrix& matrix::copy_by_row(const matrix& mat, int startrow, int count)
{
	assert(col == mat.col);
	memcpy(val, mat.val + startrow * col, sizeof(float) * count * col);

	return *this;
}

matrix& matrix::copy_from_array(const float* arr, int len)
{
	assert(col * row == len);
	memcpy(val, arr, sizeof(float) * len);
	return *this;
}

void matrix::print(const char* line) const
{
	if (line)
		std::cout << line << std::endl;

	for (int i = 0; i < row; ++i) {
		for (int j = 0; j < col; ++j)
			std::cout << ' ' << get_val(i, j);
		std::cout << std::endl;
	}
}

#ifdef __TEST_MATRIX__
#undef __TEST_MATRIX__

int main()
{
	int row = 3;
	int col = 2;
	float data[] = {
		1.0f, 1.0f, 
		1.0f, 2.0f, 
		2.0f, 3.0f
	};

	float data2[] = {
		100.0f, 100.0f, 
		200.f, 200.f, 
		300.f, 300.f
	};

	matrix mat(data, row, col);
	matrix mat2(data2, row, col);

	matrix* pmat = 0;
	matrix::transpose(&pmat, mat);
	std::cout << "mat:" << std::endl;
	mat.print();
	std::cout << "pmat:" << std::endl;
	pmat->print();

	std::cout << "mat2:" << std::endl;
	mat2.print();

	matrix* paddsum = 0;
	std::cout << "mat2 + mat(static):" << std::endl;
	matrix::plus(&paddsum, mat, mat2)->print();
	delete paddsum;
	paddsum = 0;

	mat2.plus(mat);
	std::cout << "mat2 += mat:" << std::endl;
	mat2.print();

	std::cout << "mat2 -= mat(static):" << std::endl;
	matrix::minus(&paddsum, mat2, mat)->print();
	delete paddsum;
	paddsum = 0;

	mat2.minus(mat);
	std::cout << "mat2 -= mat:" << std::endl;
	mat2.print();
	
	mat.transpose();
	std::cout << "mat transposed:" << std::endl;
	mat.print();

	std::cout << "mat2 * mat:" << std::endl;
	matrix::multiply(&paddsum, mat2, mat)->print();
	delete paddsum;
	paddsum = 0;
}

#endif
