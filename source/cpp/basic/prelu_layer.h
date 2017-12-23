/*
 * =====================================================================================
 *
 *       Filename:  prelu_layer.h
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

#ifndef __prelu_layer__
#define __prelu_layer__

#include "matrix0.h"
#include "net_layer.h"

class prelu_layer : public net_layer {
	private:
		int batchsize;
		int dim;
		matrix* p;
		float lambda;

		matrix* x;
		matrix* dx;
		matrix* y;
		matrix* dy;
		matrix* dp;
	public:
		prelu_layer(matrix* x, matrix* y, matrix* dx, matrix* dy, float lambda=0.1f);
		~prelu_layer();
		void eval();
		void bp();
		void stat(const char* str);
		inline matrix*& get_y() { return y; }
		inline matrix*& get_dy() { return dy; }
		inline matrix*& get_x() { return x; }
		inline matrix*& get_dx() { return dx; }
		inline void set_lambda(float lambda) { this->lambda = lambda; }
		inline float get_lambda() { return lambda; }
};

#endif

