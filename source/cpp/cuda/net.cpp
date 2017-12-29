/*
 * =====================================================================================
 *
 *       Filename:  net.cpp
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

#include "net.h"
#include "net_layer.h"
#include "fc_layer.h"
#include "sigmoid_layer.h"
#include "tanh_layer.h"
#include "relu_layer.h"
#include "prelu_layer.h"
#include "softplus_layer.h"
#include "batch_normalize.h"
#include "cnn.h"
#include "maxpooling.h"
#include "avgpooling.h"
#include "dropout.h"
#include <algorithm>
#include <vector>
#include <iterator>
#include <cassert>
#include <iostream>

using std::vector;

net::net(int batch_size, int input_width) : 
	layers(), x(), dx(), output_width()
	, expected(0), batch_size(batch_size), input_width(input_width)
	, noise(false), noise_prob(0.01)
	, bns(), fcs(), drops()
{
}

net::~net()
{
	std::for_each(std::begin(layers), std::end(layers), [](net_layer* p){ delete p; });
	std::for_each(std::begin(x), std::end(x), [](matrix* p){ delete p; });
	std::for_each(std::begin(dx), std::end(dx), [](matrix* p){ delete p; });
	if (expected) delete expected;
}

void net::init()
{
	matrix* _x= new matrix(batch_size, input_width);
	matrix* _dx= new matrix(batch_size, input_width);
	x.push_back(_x);
	dx.push_back(_dx);
}

void net::finish()
{
	if (expected) {
		delete expected;
		expected = 0;
	}
	expected = new matrix(batch_size, output_width[output_width.size() - 1]);
}

void net::add_fc(int width, float lambda)
{
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);

	net_layer* layer = new fc_layer(*x.rbegin(), y, *dx.rbegin(), dy, lambda);

	push_back(y, dy, layer, width);

	fcs.push_back(layers.size() - 1);
}

void net::add_sigmoid(int width)
{
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);

	net_layer* layer = new sigmoid_layer(*x.rbegin(), y, *dx.rbegin(), dy);

	push_back(y, dy, layer, width);
}

void net::add_tanh(int width)
{
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);

	net_layer* layer = new tanh_layer(*x.rbegin(), y, *dx.rbegin(), dy);

	push_back(y, dy, layer, width);
}

void net::add_relu(int width)
{
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);

	net_layer* layer = new relu_layer(*x.rbegin(), y, *dx.rbegin(), dy);

	push_back(y, dy, layer, width);
}

void net::add_prelu(int width, float lambda)
{
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);

	net_layer* layer = new prelu_layer(*x.rbegin(), y, *dx.rbegin(), dy, lambda);

	push_back(y, dy, layer, width);
}

void net::add_softplus(int width)
{
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);

	net_layer* layer = new softplus_layer(*x.rbegin(), y, *dx.rbegin(), dy);

	push_back(y, dy, layer, width);
}

void net::add_bn(int width, float lambda)
{
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);

	net_layer* layer = new batch_normalize(*x.rbegin(), y, *dx.rbegin(), dy, 0.1f);

	push_back(y, dy, layer, width);

	bns.push_back(layers.size() - 1);
}

void net::add_cnn(int xd, int xr, int xc, int nf, int fr, int fc, float lambda)
{
	matrix* y = new matrix(batch_size, nf * xr * xc);
	matrix* dy = new matrix(batch_size, nf * xr * xc);
	net_layer* layer = new cnn(*x.rbegin(), y, *dx.rbegin(), dy, xd, xr, xc, nf, fr, fc, batch_size, lambda);
	push_back(y, dy, layer, nf * xr * xc);
}

void net::add_mp(int xd, int xr, int xc, int rstep, int cstep)
{
	int width = xd * xr * xc / rstep / cstep;
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);
	net_layer* layer = new maxpooling(*x.rbegin(), y, *dx.rbegin(), dy, xd, xr, xc, rstep, cstep);
	push_back(y, dy, layer, width);
}

void net::add_ap(int xd, int xr, int xc, int rstep, int cstep)
{
	int width = xd * xr * xc / rstep / cstep;
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);
	net_layer* layer = new avgpooling(*x.rbegin(), y, *dx.rbegin(), dy, xd, xr, xc, rstep, cstep);
	push_back(y, dy, layer, width);
}

void net::add_drop(float prob)
{
	matrix* mx = *x.rbegin();
	matrix* mdx = *dx.rbegin();
	int width = mx->get_col();
	matrix* y = new matrix(batch_size, width);
	matrix* dy = new matrix(batch_size, width);
	net_layer* layer = new dropout(mx, y, mdx, dy, prob);
	push_back(y, dy, layer, width);
	drops.push_back(layers.size() - 1);
}

void net::push_back(matrix* x, matrix* dx, net_layer* layer, int width)
{
	this->x.push_back(x);
	this->dx.push_back(dx);
	this->layers.push_back(layer);
	this->output_width.push_back(width);
}

void net::fill_in_x(const float* x, int len)
{
	matrix* input = this->x[0];
	input->copy_from_array(x, len);

	if(noise) {
		add_noise();
	}
}

void net::fill_in_y(const float* y, int len)
{
	expected->copy_from_array(y, len);
}

void net::eval(const bool train)
{
	std::for_each(std::begin(layers), std::end(layers), [=](net_layer* layer){ layer->eval(train); });
}
	
void net::bp()
{
	std::for_each(layers.rbegin(), layers.rend(), [](net_layer* layer){ layer->bp(); });
}


void net::begin_collect(const vector<int>& bns)
{
	std::for_each(std::begin(bns), std::end(bns),
			[=](int sn){ (dynamic_cast<batch_normalize*>(layers[sn]))->begin_collect(); });
}

void net::collect(const vector<int>& bns)
{
	std::for_each(std::begin(bns), std::end(bns),
			[=](int sn){ (dynamic_cast<batch_normalize*>(layers[sn]))->collect(); });
}

void net::end_collect(const vector<int>& bns)
{
	std::for_each(std::begin(bns), std::end(bns),
			[=](int sn){ (dynamic_cast<batch_normalize*>(layers[sn]))->end_collect(); });
}

void net::collapse_bn_to_fc(const vector<int>& bns, const vector<int>& fcs)
{
	assert(bns.size() == fcs.size());
	for (int i = 0; i < bns.size(); ++i) {
		assert(bns[i] == fcs[i] + 1);
		(dynamic_cast<batch_normalize*>(layers[bns[i]]))->collapse_to_fc(
				dynamic_cast<fc_layer&>(*layers[fcs[i]]));
		layers[fcs[i]]->get_y() = layers[bns[i]]->get_y();
		layers[fcs[i]]->get_dy() = layers[bns[i]]->get_dy();
	}

    int dropi = drops.size() - 1;
	for (int i = bns.size() - 1; i >= 0; --i) {
        while (dropi >= 0 && drops[dropi] > bns[i]) {
            drops[dropi] -= (i + 1);
            dropi--;
        }
		delete x[bns[i]];
		x.erase(x.begin() + bns[i]);
		delete dx[bns[i]];
		dx.erase(dx.begin() + bns[i]);

		delete layers[bns[i]];
		layers.erase(layers.begin() + bns[i]);

		output_width.erase(output_width.begin() + bns[i]);
	}
}

void net::remove_drops()
{
    int bni = bns.size() - 1;
    int fci = fcs.size() - 1;
	for(int i = drops.size() - 1; i >= 0; --i) {
		int sn = drops[i];

        while (bni >= 0 && bns[bni] > sn) {
            bns[bni] -= (i + 1);
            bni--;
        }
        while (fci >= 0 && fcs[fci] > sn) {
            fcs[fci] -= (i + 1);
            fci--;
        }

		layers[sn + 1]->get_x() = layers[sn]->get_x();
		layers[sn + 1]->get_dx() = layers[sn]->get_dx();

		delete x[sn + 1];
		x.erase(x.begin() + sn + 1);
		delete dx[sn + 1];
		dx.erase(dx.begin() + sn + 1);

		delete layers[sn];
		layers.erase(layers.begin() + sn);

		output_width.erase(output_width.begin() + sn);
	}
}

void net::set_lambda(float lambda)
{
	for (int i = 0; i < layers.size(); ++i) {
		layers[i]->set_lambda(lambda);
	}
}

void net::eval(float& accu, float& loss, const matrix* x, const matrix* y)
{
	epoch(accu, loss, x, y, false);
}

void net::epoch(float& accu, float& loss, const matrix* x, const matrix* y)
{
	epoch(accu, loss, x, y, true);
}

void net::bn_post_handle(const matrix* x, const matrix* y)
{
	float accu, loss;
	epoch(accu, loss, x, y, false, true);

	vector<int> to_collapse_bn;
	vector<int> to_collapse_fc;
	int bni = 0;
	int fci = 0;
	while (bni < bns.size() && fci < fcs.size()) {
		if (bns[bni] == fcs[fci] + 1) {
			to_collapse_bn.push_back(bns[bni]);
			to_collapse_fc.push_back(fcs[fci]);
			++bni; ++fci;
		} else if (bns[bni] < fcs[fci]) {
			++bni;
		} else {
			++fci;
		}
	}
	collapse_bn_to_fc(to_collapse_bn, to_collapse_fc);
}

void net::epoch(float& accu, float& loss, const matrix* x, const matrix* y
		, const bool train, const bool collect)
{
	int xtotalsize = x->get_row() * x->get_col();
	int batch_xdata_len = batch_size * x->get_col();
	int batch_ydata_len = batch_size * y->get_col();

	int startxpos = 0;
	int startypos = 0;

	loss = 0.0f;
	accu = 0;
	int batch_count = 0;
	if (collect) begin_collect(bns);

	do {
		float batch_loss = 0;
		float batch_accu = 0;
		batch(batch_loss, batch_accu, x->val + startxpos, batch_xdata_len
				, y->val + startypos, batch_ydata_len, train);
		if (collect) net::collect(bns);

		loss += batch_loss;
		accu += batch_accu;

		startxpos += batch_xdata_len;
		startypos += batch_ydata_len;
		++batch_count;
	} while(startxpos < xtotalsize);
	if (collect) end_collect(bns);

	loss /= batch_count;
	accu /= batch_count;
}

void net::batch(float& loss_val, float& accu_val
		, const float* x, const int xlen, const float* y, const int ylen, const bool train)
{
	fill_in_x(x, xlen);
	fill_in_y(y, ylen);

	eval(train);
	loss_val = loss();
	accu_val = accuracy();

	if (train) {
		dloss();
		bp();
	}
}

void net::dump_output()
{
	std::cout << layers.size() << std::endl;
	std::cout << x.size() << std::endl;
	std::cout << dx.size() << std::endl;
	std::cout << output_width.size() << std::endl;
	/*  
	for (int i = 0; i < layers.size() - 2; i += 2) {
		std::cout << "assert layer i : " << i << " to " << i + 1 << std::endl;
		assert(dynamic_cast<fc_layer*>(layers[i]));
		assert(dynamic_cast<sigmoid_layer*>(layers[i + 1]));
		assert(dynamic_cast<fc_layer*>(layers[i])->get_y() == dynamic_cast<sigmoid_layer*>(layers[i + 1])->get_x());
		assert(dynamic_cast<fc_layer*>(layers[i])->get_dy() == dynamic_cast<sigmoid_layer*>(layers[i + 1])->get_dx());
		assert(dynamic_cast<fc_layer*>(layers[i])->get_y() == x[i + 1]);
		assert(dynamic_cast<fc_layer*>(layers[i])->get_dy() == dx[i + 1]);
		assert(dynamic_cast<fc_layer*>(layers[i])->get_x() == x[i]);
		assert(dynamic_cast<fc_layer*>(layers[i])->get_dx() == dx[i]);
		assert(dynamic_cast<sigmoid_layer*>(layers[i + 1])->get_y() == x[i+2]);
		assert(dynamic_cast<sigmoid_layer*>(layers[i + 1])->get_dy() == dx[i+2]);
	}
	  */

//	matrix* output = x[x.size() - 1];
//	output->print();
}
