/*
 * =====================================================================================
 *
 *       Filename:  dropout.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/18 17时00分13秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __dropout__
#define __dropout__

#include "matrix0.h"
#include "net_layer.h"

class dropout : public net_layer {
	public:
		matrix* x;
		matrix* y;
		matrix* dx;
		matrix* dy;
		matrix* mask;
		float prob;

	public:
		dropout(matrix* x, matrix* y, matrix* dx, matrix* dy, float prob);
		~dropout();
		void eval(const bool train);
		void bp();
		void stat(const char* str);
		inline matrix*& get_y() { return y; }
		inline matrix*& get_dy() { return dy; }
		inline matrix*& get_x() { return x; }
		inline matrix*& get_dx() { return dx; }
};

#endif
