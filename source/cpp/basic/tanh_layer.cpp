/*
 * =====================================================================================
 *
 *       Filename:  tanh_layer.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/27 08时33分23秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "tanh_layer.h"
#include "ele_op.h"
#include <cassert>
#include <cmath>
#include <iostream>

tanh_layer::tanh_layer(matrix* x, matrix* y, matrix* dx, matrix* dy) :
	batchsize(x->get_row()), dim(x->get_col()), x(x), dx(dx), y(y), dy(dy)
{
	assert(x->get_row() == y->get_row());
	assert(x->get_col() == y->get_col());
}

tanh_layer::~tanh_layer()
{

}

void tanh_layer::eval()
{
	ele_op::tanh(&y, *x);
}

void tanh_layer::bp()
{
	// y->print("xxxxxxxxxxxx");
	ele_op::grad_tanh(&dx, *y);
	ele_op::multiply(*dx, *dy);
}

void tanh_layer::stat(const char* head)
{
	std::cout << head << std::endl;
	int count0 = 0;
	int count1 = 0;
	float total = y->get_row() * y->get_col();
	const float* val = y->get_val();
	for (int i = 0; i < y->get_row() * y->get_col(); ++i)
		if (val[i] < -0.999f)
			++count0;
		else if (val[i] > 0.999f)
			++count1;
		else
			;

	std::cout << "less than -0.999f : " << count0 << ", " << count0 * 100 / total << "%." << std::endl;
	std::cout << "more than 0.999f : " << count1 << ", " << count1 * 100 / total << "%." << std::endl;
}

#ifdef __TEST_TANH_LAYER__
#undef __TEST_TANH_LAYER__
int main(int argc, char** argv)
{
	srand(time(0));
	float x[] = {
		1.0, 2.0, 3.0
	};

	int len = sizeof(x) / sizeof(float);
	float* y = new float[len];

	float* dx = new float[len];

	float dy[] = {
		2.0, 2.0, 2.0
	};

	matrix mx(x, 1, 3);
	matrix my(y, 1, 3);
	matrix mdx(dx, 1, 3);
	matrix mdy(dy, 1, 3);
	tanh_layer tanh(&mx, &my, &mdx, &mdy);
	tanh.eval();
	tanh.bp();

	mx.print("x:");
	my.print("y:");
	mdx.print("dx:");
	mdy.print("dy:");
	return 0;
}

#endif
