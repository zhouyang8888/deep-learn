/*
 * =====================================================================================
 *
 *       Filename:  tensor.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/24 11时15分14秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "tensor.h"
#include <cstring>
#include <iostream>

tensor::tensor(int size) : size(size)
{
	val = new float[size];
	memset(val, 0, sizeof(float) * size);
}

tensor::tensor(float* val, int size) : size(size)
{
	this->val = new float[size];
	memcpy(this->val, val, sizeof(float) * size);
}

tensor::tensor(const tensor& other) : size(other.size)
{
	this->val = new float[size];
	memcpy(this->val, other.val, sizeof(float) * size);
}
tensor::~tensor()
{
	if(val) {
		delete[] val;
	}
}

void tensor::plus_num(const float b)
{
	for(int i = 0; i < size; ++i)
		val[i] += b;
}

void tensor::plus_num(tensor& ret, const float b) const
{
	for(int i = 0; i < size; ++i)
		ret.val[i] = val[i] + b;
}

void tensor::minus_num(const float b)
{
	for(int i = 0; i < size; ++i)
		val[i] -= b;
}

void tensor::minus_num(tensor& ret, const float b) const
{
	for(int i = 0; i < size; ++i)
		ret.val[i] = val[i] - b;
}

void tensor::multiply_num(const float b)
{
	for(int i = 0; i < size; ++i)
		val[i] *= b;
}
void tensor::multiply_num(tensor& ret, const float b) const
{
	for(int i = 0; i < size; ++i)
		ret.val[i] = val[i] * b;
}

void tensor::divide_num(const float b)
{
	for(int i = 0; i < size; ++i)
		val[i] /= b;
}

void tensor::divide_num(tensor& ret, const float b) const
{
	for(int i = 0; i < size; ++i)
		ret.val[i] = val[i] / b;
}

void tensor::plus_tensor(const tensor& b)
{
	for(int i = 0; i < size; ++i)
		val[i] += b.val[i];
}

void tensor::plus_tensor(tensor& ret, const tensor& b) const
{
	for(int i = 0; i < size; ++i)
		ret.val[i] = val[i] + b.val[i];
}

void tensor::minus_tensor(const tensor& b)
{
	for(int i = 0; i < size; ++i)
		val[i] -= b.val[i];
}

void tensor::minus_tensor(tensor& ret, const tensor& b) const
{
	for(int i = 0; i < size; ++i)
		ret.val[i] = val[i] - b.val[i];
}

float tensor::dot_product(const tensor& b) const
{
	float s = 0.0f;
	for (int i = 0; i < size; ++i)
		s += val[i] * b.val[i];
	return s;
}

#ifdef __TEST_TENSOR__
int main()
{
	float vec[] = {1.0f, 1.0f};
	tensor ts(vec, 2);
	float vec2[] = {2.0f, 3.0f};
	tensor ts2(vec2, 2);
	const float* val = ts.get_val();
	int size = ts.get_size();

	ts.plus_num(10);
	for (int i = 0; i < size; ++i) 
		std::cout << val[i] << ", " << std::endl;

	ts.plus_tensor(ts2);
	for (int i = 0; i < size; ++i) 
		std::cout << val[i] << ", " << std::endl;
	return 0;
}
#undef __TEST_TENSOR__
#endif
