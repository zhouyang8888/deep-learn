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

#include <cstdlib>
#include <iostream>
#include "linear_regression.h"
#include "gaussian.h"


linear_regression::linear_regression(float lambda, int dim) : 
	lambda(lambda), pw(0), loss(10000), 
	pm_yeval(0), pm_yevaltrans(0), pm_loss(0), pm_grad(0)
{
	gaussian gau;
	float* tmp = new float[dim];
	for (int i = 0; i < dim; ++i) tmp[i] = (float) gau.get(0, 0.01f);
	pw = new matrix(tmp, dim, 1);

	delete[] tmp;
}

linear_regression::~linear_regression()
{
	if (pw) delete pw;
	if (pm_yeval) delete pm_yeval;
	if (pm_yevaltrans) delete pm_yevaltrans;
	if (pm_loss) delete pm_loss;
	if (pm_grad) delete pm_grad;
}

matrix* linear_regression::evaluate(const matrix& xarray)
{
	matrix::multiply(&pm_yeval, xarray, *pw);
	return pm_yeval;
}

matrix* linear_regression::gradient(matrix& dy, const matrix& x)
{
	(matrix::multiply(&pm_grad, dy.transpose(), x))->transpose();
	pm_grad->divide_num(x.get_row());
	return pm_grad;
}

void linear_regression::batch_train(const matrix& xarray, const matrix& yarray)
{
	evaluate(xarray);
	
	// dy = diff
	pm_yeval->minus(yarray);

	// loss
	pm_yevaltrans = matrix::transpose(&pm_yevaltrans, *pm_yeval);
	matrix::multiply(&pm_loss, *pm_yevaltrans, *pm_yeval);
	float lossval = pm_loss->get_val(0, 0) / (2 * xarray.get_row());
	if (lossval > loss) lambda /= 1.2f;
	loss = lossval;
	
	// gradient
	gradient(*pm_yeval, xarray);

	// update
	pw->minus(pm_grad->multiply_num(lambda));
}

int main()
{
	const int row = 24;
	const int col = 1;
	const int mid = 2;
	float w[] = {12.5f, -3.0f};
	float x[] = {
		0, 1, 1, 1, 2, 1, 10, 1, 
			3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 10, 1, 
		0, 1, 1, 1, 2, 1, 10, 1, 
			3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9, 1, 10, 1, 
	};
	matrix xarray(x, row, mid);
	matrix standardw(w, mid, 1);
	matrix* yarray = 0;
	matrix::multiply(&yarray, xarray, standardw);

	std::cout << "standard y " << std::endl;
	yarray->print();

	gaussian gau;
	for (int i = 0; i < row; ++i)
		yarray->set_val(i, 0, yarray->get_val(i, 0) + gau.get(0, 0.5f));

	std::cout << "xarray " << std::endl;
	xarray.print();
	std::cout << "~~~~yarray " << std::endl;
	yarray->print();

	linear_regression lr(0.1f, mid);
	int loop = 0;
	while (loop < 5000000) {
		lr.batch_train(xarray, *yarray);
		++loop;

		/* 
		std::cout << "loop " << loop << std::endl;
		const matrix* pw =lr.get_coeffients();
		pw->print();

		std::cout << "lambda : " << lr.get_lambda() 
			<< " loss : " << lr.get_loss() << std::endl;
		std::cout << "###########################" << std::endl;
		*/
	}

	{
		std::cout << "loop " << loop << std::endl;
		const matrix* pw =lr.get_coeffients();
		pw->print();

		std::cout << "lambda : " << lr.get_lambda() 
			<< " loss : " << lr.get_loss() << std::endl;
		std::cout << "###########################" << std::endl;
	}

	std::cout << "expected w : " << std::endl;
	for (int i = 0; i < sizeof(w) / sizeof(float); ++i)
		std::cout << w[i] << " ";
	std::cout << std::endl;
}
