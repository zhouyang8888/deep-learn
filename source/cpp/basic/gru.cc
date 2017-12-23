/*
 * =====================================================================================
 *
 *       Filename:  gru.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/21 23时17分15秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef __gru__
#define __gru__

#include "matrix0.h"
#include "net_layer.h"

gru::gru(matrix* x, matrix* ht_1, matrix* ht, int len, int xwidth, int hwidth, float lambda);
class gru : net_layer {
	public:
		matrix* x;    // m
		matrix* h1;   // n
		matrix* h;    // n
		matrix* y;    // output y==h : n 

		int xwidth;   // = m
		int hwidth;   // = n
		matrix* h1_f;    // ht_1 filtered: n
		matrix* r;       // control forgetting of ht_1 : n
		matrix* z;       // control new status update and output : n
		matrix* h_hat;   // new added ht_hat delta : n

		matrix* rxw;     // rx linear weights : m * n
		matrix* rhw;     // rh linear weights : n * n
		matrix* rb;      // rb bias : n

		matrix* zxw;     // zt linear weights : m * n
		matrix* zhw;     // zt linear weights : n * n
		matrix* zb;      // zb bias : n

		matrix* hxw;     // hx linear weights : m * n
		matrix* hhw;     // hhf linear weights : n * n
		matrix* hb;      // hb bias : n


		//////////////////////////////////////diff
		//
		matrix* dx;
		matrix* dh1;
		matrix* dh;
		matrix* dy;  

		matrix* dh1_f;
		matrix* dr;
		matrix* dz;
		matrix* dh_hat;

		matrix* drxw;
		matrix* drhw;
		matrix* drb;

		matrix* dzxw;
		matrix* dzhw;
		matrix* dzb;  

		matrix* dhxw;  
		matrix* dhhw;  
		matrix* dhb;  

		matrix lambda;
	public:
		gru(matrix* x, matrix* ht_1, matrix* ht, int len, int xwidth, int hwidth, float lambda);
		~gru();

		void eval();
		void bp();
		void stat(const char* str);

		inline virtual void set_lambda(float lambda) {}
		inline virtual float get_lambda() { return 0.f; }
		inline virtual void decay_lambda(float v) { set_lambda(v * get_lambda()); }
	private:
		void e_sigmoid(matrix* y, const matrix* x);
		void e_tanh(matrix* y, const matrix* x);
		void e_linear(matrix* y, const matrix* x, const matrix* w);

		void d_sigmoid(matrix* dx, const matrix* dy, const matrix* y);
		void d_tanh(matrix* dx, const matrix* dy, const matrix* y);
		void d_dotproduct(matrix* dw, const matrix* dy, const matrix* x);
};

#endif
