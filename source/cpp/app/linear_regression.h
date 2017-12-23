/*
 * =====================================================================================
 *
 *       Filename:  linear_regression.c
 *
 *    Description:  y = f(x) = tranpose(theta) * x, where x is expanded with an extra constant 1, 
 *    				for bias coeffient b.
 *
 *        Version:  1.0
 *        Created:  2017/11/24 07时37分30秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  zhouyang (blasterzhou@163.com, blasterzhou@gmail.com)
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __linear_regression__
#define __linear_regression__
#include "matrix0.h"

class linear_regression {
	private:
		float lambda;
		float loss;
		matrix* pw;

	public:
		linear_regression(float lambda, int dim);
		~linear_regression();

		matrix* evaluate(const matrix& xarray);
		matrix* gradient(matrix& dy, const matrix& x);
		void batch_train(const matrix& xarray, const matrix& yarray);

		inline const matrix* get_coeffients() const { return pw; }
		inline float get_lambda() { return lambda; }
		inline float get_loss() { return loss; }

	private:
		matrix* pm_yeval;
		matrix* pm_yevaltrans;
		matrix* pm_loss;
		matrix* pm_grad;
};

#endif
