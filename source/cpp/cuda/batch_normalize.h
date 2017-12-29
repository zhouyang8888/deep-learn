/*
 * =====================================================================================
 *
 *       Filename:  batch_normalize.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/01 18时00分27秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __BATCH_NORMALIZE__
#define __BATCH_NORMALIZE__

#include "matrix0.h"
#include "net_layer.h"
#include "fc_layer.h"
#include "rmsprop.h"
#include "adam.h"
#include "momentum.h"

class batch_normalize : public net_layer {
	private:
		int m;
		int f;
		matrix* x;     // m * f
		matrix* mu;    // 1 * f
		matrix* var;   // 1 * f
		matrix* xhat;  // m * f
		matrix* gamma; // 1 * f
		matrix* beta;  // 1 * f
		matrix* y;     // m * f
		// bp grad
		matrix* dx;         // m * f
		matrix* dy;         // m * f
		matrix* dgamma;     // 1 * f
		matrix* dbeta;      // 1 * f
		matrix* dy_dxhat;   // 1 * f
		matrix* dxhat_dvar; // m * f
		matrix* dxhat_dmu;  // 1 * f
		matrix* dxhat_dx;   // 1 * f
		float dvar_dmu;     // 0
		matrix* dvar_dx;    // m * f
		float dmu_dx;       // 1 / m
		// step
		float lambda;

		// gradient descent
		momentum* gd_m_g;
		momentum* gd_m_b;
		rmsprop* gd_r_g;
		rmsprop* gd_r_b;
		adam* gd_a_g;
		adam* gd_a_b;
	private:
		// for collect mu and var values.
		matrix* col_mu;
		matrix* col_var;
		int batch_count;
	public:
		batch_normalize(matrix* x, matrix* y, matrix* dx, matrix* dy, float lambda=0.1f);
		~batch_normalize();
		void eval(const bool train);
		void grad();
		void update();
		void bp();
		void stat(const char* head);
		inline matrix*& get_y() { return y; }
		inline matrix*& get_dy() { return dy; }
		inline matrix*& get_x() { return x; }
		inline matrix*& get_dx() { return dx; }

		inline void set_lambda(float lambda) { this->lambda = lambda; }
		inline float get_lambda() { return lambda; }

		// collect mu and var.
		void begin_collect();
		void collect();
		void end_collect();
		void collapse_to_fc(fc_layer& fc);
	private:
		void op_dgamma();
		void op_dbeta();
		void op_dx();

		void op_dxhat_dxdmudvar();
		void op_dvar_dx();
};

#endif
