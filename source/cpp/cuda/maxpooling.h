/*
 * =====================================================================================
 *
 *       Filename:  maxpooling.h
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

#ifndef __maxpooling__
#define __maxpooling__

#include "matrix0.h"
#include "net_layer.h"

class maxpooling : public net_layer {
	public:
		matrix* x;
		matrix* y;
		matrix* dx;
		matrix* dy;

		int xd;
		int xr;
		int xc;
		int rstep;
		int cstep;

		int xarea;
		int ywidth;
		int yheight;
		int yarea;
	public:
		maxpooling(matrix* x, matrix* y, matrix* dx, matrix* dy
				, int xd, int xr, int xc, int rstep, int cstep);
		~maxpooling();
		void eval(const bool train);
		void bp();
		void stat(const char* str);
		inline matrix*& get_y() { return y; }
		inline matrix*& get_dy() { return dy; }
		inline matrix*& get_x() { return x; }
		inline matrix*& get_dx() { return dx; }
};

#endif
