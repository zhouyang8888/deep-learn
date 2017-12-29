/*
 * =====================================================================================
 *
 *       Filename:  cnn.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/13 15时13分14秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __cnn__
#define __cnn__

#include "cube.h"
#include "matrix0.h"
#include "net_layer.h"
#include "momentum.h"
#include "rmsprop.h"
#include "adam.h"

class cnn : public net_layer {
	public:
		matrix* mx;
		matrix* my;
		matrix* dmx;
		matrix* dmy;

		vector<cube> x;
		vector<cube> y;
		vector<cube> f;
		cube b;
		cube db;
		int bz;
		int xd;
		int xr;
		int xc;
		int nf;
		int fd;
		int fr;
		int fc;
		int yd;
		int yr;
		int yc;

		vector<cube> dx;
		vector<cube> dy;
		vector<cube> df;
		float lambda;
	protected:
		momentum* gd_m_w;
		momentum* gd_m_b;
		rmsprop* gd_r_w;
		rmsprop* gd_r_b;
		adam* gd_a_w;
		adam* gd_a_b;

	public:

		cnn(matrix* x, matrix* y, matrix* dx, matrix* dy
				, int xd, int xr, int xc 
				, int nf, int fr, int fc
				, int bz, float lambda=0.1f);
		~cnn();
		void eval(const bool train);
		void bp();
		void stat(const char* str);
		inline matrix*& get_y() { return my; }
		inline matrix*& get_dy() { return dmy; }
		inline matrix*& get_x() { return mx; }
		inline matrix*& get_dx() { return dmx; }
		inline void set_lambda(float lambda) { this->lambda = lambda; }
		inline float get_lambda() { return lambda; }
	private:
		void difff();
		void diffx();
		void diffb();
		void flatten(matrix* mx, vector<cube>& x, int depth);
		void to3d(matrix* mx, vector<cube>& x, int depth);
		void slip(cube& x, cube& f, vector<vector<float> >& y, vector<vector<float> >& b);
};

#endif
