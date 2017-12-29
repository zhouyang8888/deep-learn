/*
 * =====================================================================================
 *
 *       Filename:  adam.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/08 13时40分54秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __adam__
#define __adam__
#include "matrix0.h"

class adam {
	private:
		matrix* s;
		matrix* r;
		float rho1;
		float rho2;
		float rho1_power;
		float rho2_power;
	public:
		adam(int row, int col, float rho1=0.9f, float rho2=0.999f);
		~adam();

		void update(matrix* theta, matrix* g, float lambda);
};

#endif

