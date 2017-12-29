/*
 * =====================================================================================
 *
 *       Filename:  ele_op.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/28 10时11分34秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __ele_op__
#define __ele_op__
#include "matrix0.h"
#include "gaussian.h"

class ele_op {
	public:
		static matrix& sigmoid(matrix& mat);
		static matrix* sigmoid(matrix** ret, const matrix& mat);
		static matrix& grad_sigmoid(matrix& mat);
		static matrix* grad_sigmoid(matrix** ret, const matrix& mat);
		
		static matrix* relu(matrix** ret, const matrix& x);
		static matrix* grad_relu(matrix** ret, const matrix& x);

		static matrix* softplus(matrix** ret, const matrix& x);
		static matrix* grad_softplus(matrix** ret, const matrix& x);

		static matrix* prelu(matrix** ret, const matrix& x, const matrix& pp);
		static matrix* grad_prelu(matrix** ret, matrix& dp, const matrix& x, const matrix& p);

		static matrix& tanh(matrix& mat);
		static matrix* tanh(matrix** ret, const matrix& mat);
		static matrix& grad_tanh(matrix& mat);
		static matrix* grad_tanh(matrix** ret, const matrix& mat);

		static matrix& nearest_int(matrix& mat);
		static matrix* nearest_int(matrix** ret, const matrix& mat);

		static matrix& gaussian_init(matrix& mat, gaussian& gau, float mean=0.f, float stdvar=1.f);

		static matrix& multiply(matrix& ret, const matrix& mat);
		static matrix* multiply(matrix** ret, const matrix& mat0, const matrix& mat1);

		static matrix& sum_row_vectors(matrix& ret, const matrix& mat);
		static matrix& mean_row_vectors(matrix& ret, const matrix& mat);
		static matrix& variance_row_vectors(matrix& ret, const matrix& mean, const matrix& mat);
		static matrix& normalize_row_vectors(matrix& xhat, const matrix& mean, const matrix& var, const matrix& x);
		static matrix& linear_row_vectors(matrix& xhat, const matrix& x, const matrix& gamma, const matrix& beta);

		// batch normal op : dx.
		static matrix& bn_dx(matrix& dx, const matrix& gamma, const matrix& std_var, const matrix& xhat, const matrix& dy);
		// batch normal op : dgamma.
		static matrix& bn_dgamma(matrix& dgamma, const matrix& xhat, const matrix& dy);

};

#endif
