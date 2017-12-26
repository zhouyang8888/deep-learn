/*
 * =====================================================================================
 *
 *       Filename:  net.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/30 17时05分48秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef __net__
#define __net__
#include "matrix0.h"
#include "net_layer.h"
#include <vector>
#include <algorithm>

using std::vector;

class net {
	public:
		vector<net_layer*> layers;
		vector<matrix*> x;
		vector<matrix*> dx;
		vector<int> output_width;
		matrix* expected;
		int batch_size;
		int input_width;
		bool noise;
		float noise_prob;
		virtual float loss()=0;
		virtual void dloss()=0;
		virtual float accuracy()=0;
		virtual inline void add_noise() {};

	public:
		net(int batch_size, int input_width);
		virtual ~net();
		void init();
		void add_fc(int width, float lambda);
		void add_sigmoid(int width);
		void add_tanh(int width);
		void add_relu(int width);
		void add_prelu(int width, float lambda);
		void add_softplus(int width);
		void add_bn(int width, float lambda);
		void add_cnn(int xd, int xr, int xc, int nf, int fr, int fc, float lambda);
		void add_mp(int xd, int xr, int xc, int rstep, int cstep);
		void add_ap(int xd, int xr, int xc, int rstep, int cstep);
		void add_drop(float prob);

		void finish();

		void fill_in_x(const float* x, const int len);
		void fill_in_y(const float* y, const int len);

		void eval(const bool train=true);
		void bp();

		void dump_output();

		void epoch(float& accu, float& loss, const matrix* x, const matrix* y);
		void eval(float& accu, float& loss, const matrix* x, const matrix* y);
		void bn_post_handle(const matrix* x, const matrix* y);

		inline const matrix* get_input() const {
			return x[0];
		}

		inline const matrix* get_output() const {
			return x[x.size() - 1];
		}

		void begin_collect(const vector<int>& bns);
		void collect(const vector<int>& bns);
		void end_collect(const vector<int>& bns);
		void collapse_bn_to_fc(const vector<int>& bns, const vector<int>& fcs);
		void remove_drops();

		void set_lambda(float lambda);
	private:
		void push_back(matrix* x, matrix* dx, net_layer* layer, int width);
		void batch(float& loss_val, float& accu_val
				, const float* x, const int xlen, const float* y, const int ylen, const bool train=true);
		void epoch(float& accu, float& loss, const matrix* x, const matrix* y
				, const bool train, const bool collect=false);
	private:
		vector<int> bns;
		vector<int> fcs;
		vector<int> drops;
};

#endif
