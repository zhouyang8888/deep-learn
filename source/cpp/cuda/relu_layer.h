/*
 * =====================================================================================
 *
 *       Filename:  fc_layer.h
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

#ifndef __relu_layer__
#define __relu_layer__

#include "matrix0.h"
#include "net_layer.h"

class relu_layer : public net_layer {
	private:
		int batchsize;
		int dim;

		matrix* x;
		matrix* dx;
		matrix* y;
		matrix* dy;
	public:
		relu_layer(matrix* x, matrix* y, matrix* dx, matrix* dy);
		~relu_layer();
		void eval(const bool train);
		void bp();
		void stat(const char* str);
		inline matrix*& get_y() { return y; }
		inline matrix*& get_dy() { return dy; }
		inline matrix*& get_x() { return x; }
		inline matrix*& get_dx() { return dx; }
};

#endif

