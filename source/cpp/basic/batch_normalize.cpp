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


#include "matrix0.h"
#include "gaussian.h"
#include "ele_op.h"
#include "constant.h"
#include "batch_normalize.h"
#include <iostream>
#include <time.h>
#include <cmath>

batch_normalize::batch_normalize(matrix* x, matrix* y, matrix* dx, matrix* dy, float lambda) : 
	m(x->get_row()), f(x->get_col()), x(x), y(y), dx(dx), dy(dy), lambda(lambda), 
	gd_m_g(0), gd_m_b(0), gd_r_g(0), gd_r_b(0), gd_a_g(0), gd_a_b(0)
{
	// eval
	mu = new matrix(1, f);
	var = new matrix(1, f);
	xhat = new matrix(m, f);
	gamma = new matrix(1, f);
	beta = new matrix(1, f);
	
	// bp grad
	dgamma = new matrix(1, f);
	dbeta = new matrix(1, f);
	dy_dxhat = gamma;   // 1 * f
	dxhat_dvar = new matrix(m, f); // m * f
	dxhat_dmu = new matrix(1, f);  // 1 * f
	dxhat_dx = new matrix(1, f);   // 1 * f
	dvar_dmu = 0;     // 0
	dvar_dx = new matrix(m, f);    // m * f
	dmu_dx = 1.0f / m;       // 1 / m

	// mu and var collection
	col_mu = 0;
	col_var = 0;
	batch_count = 0;

	// init gamma
	gaussian gau;
	ele_op::gaussian_init(*gamma, gau, 0, 1);

	// gradient descent
	gd_m_g = new momentum(1, f);
	gd_m_b = new momentum(1, f);
	gd_r_g = new rmsprop(1, f);
	gd_r_b = new rmsprop(1, f);
	gd_a_g = new adam(1, f);
	gd_a_b = new adam(1, f);
}

batch_normalize::~batch_normalize()
{
	if (mu) delete mu;
	if (var)  delete var;
	if (xhat) delete xhat;
	if (gamma) delete gamma;
	if (beta) delete beta;
	if (dgamma) delete dgamma;
	if (dbeta) delete dbeta;
	if (dxhat_dvar) delete dxhat_dvar;
	if (dxhat_dmu) delete dxhat_dmu;
	if (dxhat_dx) delete dxhat_dx;
	if (dvar_dx) delete dvar_dx;

	if (col_mu) delete col_mu;
	if (col_var) delete col_var;

	// gradient descent
	delete gd_m_g;
	delete gd_m_b;
	delete gd_r_g;
	delete gd_r_b;
	delete gd_a_g;
	delete gd_a_b;
}

void batch_normalize::eval()
{
	ele_op::mean_row_vectors(*mu, *x);
	ele_op::variance_row_vectors(*var, *mu, *x);
	// mu->print("mu:");
	// var->print("var:");
	ele_op::normalize_row_vectors(*xhat, *mu, *var, *x);
	ele_op::linear_row_vectors(*y, *xhat, *gamma, *beta);
}

void batch_normalize::grad()
{
	op_dgamma();
	op_dbeta();

	op_dxhat_dxdmudvar();
	op_dvar_dx();

	op_dx();
}

void batch_normalize::bp()
{
	grad();
	update();
}

void batch_normalize::op_dgamma()
{
	ele_op::bn_dgamma(*dgamma, *xhat, *dy);
}

void batch_normalize::op_dbeta()
{
	ele_op::sum_row_vectors(*dbeta, *dy);
}

void batch_normalize::op_dxhat_dxdmudvar()
{
	float *var = this->var->val;
	float *dxhat_dx = this->dxhat_dx->val;
	float *dxhat_dmu = this->dxhat_dmu->val;
	float *dxhat_dvar = this->dxhat_dvar->val;

	for (int j = 0, i = m * f - f; j < f; ++j, ++i) {
		dxhat_dx[j] = 1.0f / sqrt(var[j] + constant::epsilon);
		dxhat_dmu[j] = -dxhat_dx[j];

		// temporarily keep.
		dxhat_dvar[i] = (-0.5f) * dxhat_dx[j] * dxhat_dx[j] * dxhat_dx[j];
	}

	float *tmp = &dxhat_dvar[m * f - f];
	for (int i = 0, j = 0; i < m * f; ++i, ++j) {
		if (j >= f) j = 0;
		dxhat_dvar[i] = (x->val[i] - mu->val[j]) * tmp[j];
	}
}

void batch_normalize::op_dvar_dx()
{
	for (int i = 0, j = 0; i < m * f; ++i, ++j) {
		if (j >= f) j = 0;
		dvar_dx->val[i] = 2.0f * (x->val[i] - mu->val[j]) / m;
	}
}

void batch_normalize::op_dx()
{
	float *dy = this->dy->val;
	float *gamma = this->gamma->val;
	// dxhat = dy * dy_dxhat
	// dxhat in dy
	for (int i = 0, j = 0; i < m * f; ++i, ++j) {
		if (j >= f) j = 0;
		dy[i] = dy[i] * gamma[j];

		// std::cout << "dxhat (" << i / f << ", " << i % f << "): " << dy[i] << std::endl;
	}
	
	// dmu = sum_over_batch(dxhat * dxhat_dmu)
	// dmu in dxhat_dmu
	float *dxhat_dmu = this->dxhat_dmu->val;
	for (int j = 0; j < f; ++j) {
		float sum = 0.0f;
		for (int i = j; i < m * f; i += f) {
			sum += dy[i];
		}
		dxhat_dmu[j] *= sum;

		// std::cout << "dmu (" << f << "): " << dxhat_dmu[j] << std::endl;
	}

	// dvar = sum_over_batch(dxhat * dxhat_dvar)
	// dvar in dxhat_dvar, 1st row.
	float *dxhat_dvar = this->dxhat_dvar->val;
	for (int j = 0; j < f; ++j) {
		dxhat_dvar[j] *= dy[j];
		for (int i = j + f; i < m * f; i += f) {
			dxhat_dvar[j] += dxhat_dvar[i] * dy[i];
		}

		// std::cout << "dvar (" << f << "): " << dxhat_dvar[j] << std::endl;
	}

	// dx = dmu * dmu_dx + dvar * dvar_dx + dxhat * dxhat_dx
	float *dx = this->dx->val;
	float *dxhat_dx = this->dxhat_dx->val;
	float dmu_dx = this->dmu_dx;
	float *dvar_dx = this->dvar_dx->val;
	for (int i = 0, j = 0; i < m * f; ++i, ++j) {
		if (j >= f) j = 0;
		// std::cout << "dx[i] = dxhat_dmu[j] * dmu_dx + dxhat_dvar[j] * dvar_dx[i] + dy[i] * dxhat_dx[j]" << std::endl;
		// std::cout << dx[i] << "=" << dxhat_dmu[j] << "*" << dmu_dx << "+" << dxhat_dvar[j] << "*" << dvar_dx[i] << "+" << dy[i] << "*" << dxhat_dx[j] << std::endl;
		dx[i] = dxhat_dmu[j] * dmu_dx + dxhat_dvar[j] * dvar_dx[i] + dy[i] * dxhat_dx[j];

		// std::cout << "dx (" << i / f << ", " << i % f << "): " << dx[i] << std::endl;
	}
}

void batch_normalize::update()
{
	gd_m_g->update(gamma, dgamma, lambda);
	gd_m_b->update(beta, dbeta, lambda);
	// gd_r_g->update(gamma, dgamma, lambda);
	// gd_r_b->update(beta, dbeta, lambda);
	// gd_a_g->update(gamma, dgamma, lambda);
	// gd_a_b->update(beta, dbeta, lambda);
	/*  *
	gamma->minus(dgamma->multiply_num(lambda));
	beta->minus(dbeta->multiply_num(lambda));
	*  */
}

void batch_normalize::stat(const char* head)
{
	std::cout << head << std::endl;
	mu->print("mean");
	var->print("variance");
	xhat->print("xhat");
	gamma->print("gamma");
	beta->print("beta");
	dgamma->print("dgamma");
	dbeta->print("dbeta");
}

// collect mu and var.
void batch_normalize::begin_collect()
{
	if (0 == col_mu)
		col_mu = new matrix(1, f);
	else
		memset(col_mu->val, 0, sizeof(float) * 1 * f);

	if (0 == col_var)
		col_var = new matrix(1, f);
	else
		memset(col_var->val, 0, sizeof(float) * 1 * f);

	batch_count = 0;
}

void batch_normalize::collect()
{
	col_mu->plus(*mu);
	col_var->plus(*var);
	++batch_count;
}

void batch_normalize::end_collect()
{
	col_mu->divide_num(batch_count);
	col_var->multiply_num((float) m / (float) ((m - 1) * batch_count));
}

void batch_normalize::collapse_to_fc(fc_layer& fc)
{
	/* */  
	matrix* coeff = new matrix(1, f);
	matrix* bias  = new matrix(1, f);

	float* coe = coeff->val;
	float* b = bias->val;
	float* mu = this->col_mu->val;
	float* var = this->col_var->val;
	float* gamma = this->gamma->val;
	float* beta = this->beta->val;

	for (int i = 0; i < f; ++i) {
		coe[i] = gamma[i] / sqrt(var[i] + constant::epsilon);
		b[i] = beta[i] - coe[i] * mu[i];
	}

	float* fc_w = fc.w->val;
	float* fc_b = fc.b->val;

	for (int i = 0, j = 0; i < fc.xdim * fc.ydim; ++i, ++j) {
		if (j >= fc.ydim) j = 0;
		fc_w[i] *= coe[j];
		fc_b[j] = fc_b[j] * coe[j] + b[j];
	}

	delete coeff;
	delete bias;
	/*  */
}

#ifdef __TEST_BATCH_NORMALIZE__
#undef __TEST_BATCH_NORMALIZE__

int main()
{
	srand(time(0));
	float inputx[] = {
		4, 2, 3
	};
	float inputdy[] = {
		1, 2, 1
	};
	matrix *x = new matrix(inputx, 3, 1);
	matrix *y = new matrix(3, 1);
	matrix *dx = new matrix(3, 1);
	matrix *dy = new matrix(inputdy, 3, 1);
	float lambda = 0.1f;

	batch_normalize bn(x, y, dx, dy, lambda);
	/*  *
	gaussian gau;
	ele_op::gaussian_init(*x, gau, 10, 2);
	ele_op::gaussian_init(*dy, gau, 10, 2);
	*  */

	x->print("x");
	dx->print("dx");
	bn.eval();
	std::cout << "###### evaled" << std::endl;
	y->print("y");
	dy->print("dy");
	bn.dump();

	bn.bp();
	std::cout << "****** bped" << std::endl;
	dx->print("dx");
	dy->print("dy");
	bn.dump();
}

#endif
