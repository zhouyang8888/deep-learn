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

#ifndef __fc_layer__
#define __fc_layer__

#include "matrix0.h"
#include "net_layer.h"
#include "rmsprop.h"
#include "adam.h"
#include "momentum.h"

class fc_layer : public net_layer {
	friend class batch_normalize;
	protected:
		int batchsize;
		int xdim;
		int ydim;
		float lambda;

		matrix* w;
		matrix* dw;
		matrix* b;
		matrix* db;

		matrix* x;
		matrix* dx;
		matrix* y;
		matrix* dy;

		matrix* xt;
		matrix* wt;
	protected:
		momentum* gd_m_w;
		momentum* gd_m_b;
		rmsprop* gd_r_w;
		rmsprop* gd_r_b;
		adam* gd_a_w;
		adam* gd_a_b;

	public:
		fc_layer(matrix* x, matrix* y, matrix* dx, matrix* dy, float lambda = 0.01f);
		~fc_layer();
		void eval(const bool train);
		void bp();
		inline void set_lambda(float val) { lambda = val; }
		inline float get_lambda() { return lambda; }
		void dump();
		void stat(const char* head) {};
		inline matrix*& get_y() { return y; }
		inline matrix*& get_dy() { return dy; }
		inline matrix*& get_x() { return x; }
		inline matrix*& get_dx() { return dx; }
	protected:
		virtual void grad();
		void update();
	private:
		static const float GAUSSIAN_VARIANCE;
};

#endif

