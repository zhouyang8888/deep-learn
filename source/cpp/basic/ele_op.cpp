/*
 * =====================================================================================
 *
 *       Filename:  ele_op.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/28 10时15分46秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "ele_op.h"
#include "constant.h"
#include <cmath>
#include <cassert>
#include <iostream>
#include <cstring>

//////////////////sigmoid
matrix& ele_op::sigmoid(matrix& mat)
{
	for (int i = 0; i < mat.row * mat.col; ++i) {
		mat.val[i] = (float)(1.0f / (1.0f + exp(-mat.val[i])));
	}

	return mat;
}

matrix* ele_op::sigmoid(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i)
		(*ret)->val[i] = (float) (1.0f / (1.0f + expf(-mat.val[i])));

	return *ret;
}

matrix& ele_op::grad_sigmoid(matrix& mat)
{
	for (int i = 0; i < mat.row * mat.col; ++i) {
		mat.val[i] *= (1.0f - mat.val[i]);
	}

	return mat;
}

matrix* ele_op::grad_sigmoid(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i)
		(*ret)->val[i] = mat.val[i] * (1.0f - mat.val[i]);

	return *ret;
}

////////////////relu
matrix* ele_op::relu(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i)
		(*ret)->val[i] = mat.val[i] >= 0 ? mat.val[i] : 0.0f;

	return *ret;
}

// dy/dx
matrix* ele_op::grad_relu(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i)
		(*ret)->val[i] = mat.val[i] > 0 ? 1 : 0;

	return *ret;
}

//##############softplus
matrix* ele_op::softplus(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i)
		(*ret)->val[i] = log(1 + exp(mat.val[i]));

	return *ret;
}

// dy/dx
matrix* ele_op::grad_softplus(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i) {
		float ev = exp(mat.val[i]);
		(*ret)->val[i] = ev / (1 + ev);
	}

	return *ret;
}

//##############prelu
matrix* ele_op::prelu(matrix** ret, const matrix& mat, const matrix& p)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0, j = 0; i < mat.row * mat.col; ++i, ++j) {
		if (j >= mat.col) j = 0;
		(*ret)->val[i] = mat.val[i] >= 0 ? mat.val[i] : mat.val[i] * p.val[j];
	}

	return *ret;
}

// dy/dx
matrix* ele_op::grad_prelu(matrix** ret, matrix& dp, const matrix& x, const matrix& p)
{
	if (0 == *ret)
		*ret = new matrix(x.row, x.col);
	else {
		assert((*ret)->row * (*ret)->col == x.row * x.col);
		(*ret)->row = x.row;
		(*ret)->col = x.col;
	}

	memset(dp.val, sizeof(float) * dp.col, 0);
	for (int i = 0, j = 0; i < x.row * x.col; ++i, ++j) {
		if (j >= x.col) j = 0;

		if (x.val[i] >= 0) (*ret)->val[i] = 1;
		else {
			(*ret)->val[i] = p.val[j];
			dp.val[j] += x.val[i];
		}
	}

	return *ret;
}

////////////////tanh
matrix& ele_op::tanh(matrix& mat)
{
	for (int i = 0; i < mat.row * mat.col; ++i) {
		double expx = exp(mat.val[i]);
		double expnegx = 1.0 / expx;
		mat.val[i] = (float) ((expx - expnegx) / (expx + expnegx));
	}

	return mat;
}

matrix* ele_op::tanh(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i) {
		double expx = exp(mat.val[i]);
		double expnegx = 1.0 / expx;
		(*ret)->val[i] = (float)((expx - expnegx) / (expx + expnegx));
	}

	return *ret;
}

matrix& ele_op::grad_tanh(matrix& mat)
{
	for (int i = 0; i < mat.row * mat.col; ++i) {
		mat.val[i] = 1.0f - mat.val[i] * mat.val[i];
	}

	return mat;
}

matrix* ele_op::grad_tanh(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i)
		(*ret)->val[i] = 1.0f - mat.val[i] * mat.val[i];

	return *ret;
}

matrix& ele_op::nearest_int(matrix& mat)
{
	for (int i = 0; i < mat.row * mat.col; ++i) {
		int val = (int) (mat.val[i] + 0.5);
		mat.val[i] = val;
	}

	return mat;
}

matrix* ele_op::nearest_int(matrix** ret, const matrix& mat)
{
	if (0 == *ret)
		*ret = new matrix(mat.row, mat.col);
	else {
		assert((*ret)->row * (*ret)->col == mat.row * mat.col);
		(*ret)->row = mat.row;
		(*ret)->col = mat.col;
	}

	for (int i = 0; i < mat.row * mat.col; ++i) {
		int val = (int) (mat.val[i] + 0.5);
		(*ret)->val[i] = val;
	}

	return *ret;
}

///////////////////////
matrix& ele_op::gaussian_init(matrix& mat, gaussian& gau, float mean, float stdvar)
{
	for (int i = 0; i < mat.row * mat.col; ++i)
		mat.val[i] = (float) gau.get(mean, stdvar);

	return mat;
}


matrix& ele_op::sum_row_vectors(matrix& ret, const matrix& mat)
{
	assert(ret.col * ret.row == mat.col);
    ret.col = mat.col;
	ret.row = 1;
	memset(ret.val, 0, sizeof(float) * ret.col);

	for (int i = 0, j = 0; i < mat.row * mat.col; ++i, ++j) {
		if (j >= mat.col) j = 0;
		ret.val[j] += mat.val[i];
	}

	return ret;
}

matrix& ele_op::mean_row_vectors(matrix& ret, const matrix& mat)
{
	sum_row_vectors(ret, mat).divide_num(mat.get_row());

	return ret;
}

matrix& ele_op::variance_row_vectors(matrix& ret, const matrix& mean, const matrix& mat)
{
	assert(ret.col * ret.row == mat.col);
    ret.col = mat.col;
	ret.row = 1;
	memset(ret.val, 0, sizeof(float) * ret.col);

	for (int i = 0, j = 0; i < mat.row * mat.col; ++i, ++j) {
		if (j >= mat.col) j = 0;
		ret.val[j] += mat.val[i] * mat.val[i];
	}

	for (int j = 0; j < ret.col; ++j) {
		ret.val[j] = ret.val[j] / mat.row - mean.val[j] * mean.val[j];
	}

	return ret;
}

matrix& ele_op::normalize_row_vectors(matrix& ret, 
		const matrix& mean_row, const matrix& variance, const matrix& rows)
{
	assert(ret.row * ret.col == rows.row * rows.col);
	ret.row = rows.row;
	ret.col = rows.col;

	for (int i = 0, j = 0; i < rows.row * rows.col; ++i, ++j) {
		if (j >= rows.col) j = 0;
		ret.val[i] = (rows.val[i] - mean_row.val[j]) / sqrt(variance.val[j] + constant::epsilon);
	}

	return ret;
}

matrix& ele_op::linear_row_vectors(matrix& ret, const matrix& x, const matrix& gamma, const matrix& beta)
{
	assert(ret.row * ret.col == x.row * x.col);
	ret.row = x.row;
	ret.col = x.col;

	for (int i = 0, j = 0; i < x.row * x.col; ++i, ++j) {
		if (j >= x.col) j = 0;
		ret.val[i] = gamma.val[j] * x.val[i] + beta.val[j];
	}

	return ret;
}

matrix& ele_op::multiply(matrix& ret, const matrix& mat)
{
	for (int i = 0; i < ret.row * ret.col; ++i)
		ret.val[i] *= mat.val[i];
	return ret;
}

matrix* ele_op::multiply(matrix** ret, const matrix& mat0, const matrix& mat1)
{
	assert(mat0.row == mat1.row && mat0.col == mat1.col);

	if (0 == *ret)
		*ret = new matrix(mat0.row, mat0.col);
	else {
		assert((*ret)->row * (*ret)->col == mat0.row * mat0.col);
		(*ret)->row = mat0.row;
		(*ret)->col = mat0.col;
	}


	for (int i = 0; i < mat0.row * mat0.col; ++i)
		(*ret)->val[i] = mat0.val[i] * mat1.val[i];

	return *ret;
}

matrix& ele_op::bn_dx(
		matrix& dx, const matrix& gamma, const matrix& std_var, 
		const matrix& xhat, const matrix& dy)
{
	int batch_size = xhat.row;
	int dim = xhat.col;
	matrix buf(batch_size, batch_size);

	for (int i = 0; i < dim; ++i) {
		// ith feature
		float coeff = gamma.val[i] / (std_var.val[i] + constant::epsilon);
		// buf(l, k) = dy(k, i) / dx(l, i)
		for (int l = 0; l < batch_size; ++l) {
			int k = l;
			int bri = l * batch_size + k;
			int xli = l * dim + i;
			int xki = k * dim + i;
			buf.val[bri] = coeff * (1.0f - (1.0f + xhat.val[xki] * xhat.val[xli]) / batch_size);

			int bci = bri;
			while (++k < batch_size) {
				++bri;
				xki += dim;
				buf.val[bri] = coeff * (-1.0f - xhat.val[xki] * xhat.val[xli]) / batch_size;

				bci += batch_size;
				buf.val[bci] = buf.val[bri];
			}
		}

		// dloss / dx(l, i)
		for (int l = 0, i_dx = i; l < batch_size; ++l, i_dx += dim) {
			dx.val[i_dx] = 0;
			for (int k = 0, i_dy = i, i_buf = l * batch_size; 
					k < batch_size; 
					++k, i_dy += dim, ++i_buf) {
				dx.val[i_dx] += dy.val[i_dy] * buf.val[i_buf];
			}
		}
	}

	return dx;
}


matrix& ele_op::bn_dgamma(matrix& dgamma, const matrix& xhat, const matrix& dy)
{
	assert(1 == dgamma.row);
	assert(xhat.col == dgamma.col);
	assert(xhat.col == dy.col);
	assert(xhat.row == dy.row);

	for (int i = 0; i < dgamma.col; ++i) {
		dgamma.val[i] = 0;
		for (int k = 0, offset = i; k < xhat.row; ++k, offset += xhat.col)
			dgamma.val[i] += dy.val[offset] * xhat.val[offset];
	}

	return dgamma;
}
