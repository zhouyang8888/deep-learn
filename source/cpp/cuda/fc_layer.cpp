/*
 * =====================================================================================
 *
 *       Filename:  fc_layer.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/27 08时43分20秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */


#include "fc_layer.h"
#include "gaussian.h"
#include "ele_op.h"
#include <iostream>
#include <cmath>

const float fc_layer::GAUSSIAN_VARIANCE = 0.1f;

fc_layer::fc_layer(matrix* x, matrix* y, matrix* dx, matrix* dy, float lambda) : 
	batchsize(x->get_row()), xdim(x->get_col()), ydim(y->get_col()), lambda(lambda)
	, x(x), dx(dx), y(y), dy(dy) 
	, xt(0), wt(0) 
	, gd_m_w(0), gd_m_b(0), gd_r_w(0), gd_r_b(0), gd_a_w(0), gd_a_b(0)
{
	w = new matrix(xdim, ydim);
	dw = new matrix(xdim, ydim);
	b = new matrix(1, ydim);
	db = new matrix(1, ydim);

	float stdvar = 0.1f; // sqrt(4.0 / (xdim + ydim));

	gaussian gau;
	ele_op::gaussian_init(*w, gau, 0.0f, stdvar);
	// ele_op::gaussian_init(*b, gau, 0.0f, stdvar);
	
	// gradient descent.
	gd_m_w = new momentum(xdim, ydim);
	gd_m_b = new momentum(1, ydim);
	gd_r_w = new rmsprop(xdim, ydim);
	gd_r_b = new rmsprop(1, ydim);
	gd_a_w = new adam(xdim, ydim);
	gd_a_b = new adam(1, ydim);
}

fc_layer::~fc_layer()
{
	if (xt) delete xt;
	if (wt) delete wt;
	if (w)  delete w;
	if (dw) delete dw;
	if (b)  delete b;
	if (db) delete db;
	// gd
	delete gd_m_w;
	delete gd_m_b;
	delete gd_r_w;
	delete gd_r_b;
	delete gd_a_w;
	delete gd_a_b;
}

void fc_layer::eval(const bool train)
{
	matrix::multiply(&y, *x, *w);
	y->plus_row_row(*b);
	/*  
	std::cout << "x:" << std::endl;
	x->print();
	std::cout << "w:" << std::endl;
	w->print();
	std::cout << "b:" << std::endl;
	b->print();
	std::cout << "y:" << std::endl;
	y->print();
	  */
}

void fc_layer::grad()
{
	// dw
	matrix::transpose(&xt, *x);
	matrix::multiply(&dw, *xt, *dy);

	// db
	matrix::sum_rows(&db, *dy);

	// dx
	matrix::transpose(&wt, *w);
	matrix::multiply(&dx, *dy, *wt);
}

void fc_layer::bp()
{
	grad();
	update();
}

void fc_layer::update()
{
	gd_m_w->update(w, dw, lambda);
	gd_m_b->update(b, db, lambda);
	// gd_r_w->update(w, dw, lambda);
	// gd_r_b->update(b, db, lambda);
	// gd_a_w->update(w, dw, lambda);
	// gd_a_b->update(b, db, lambda);
	/*  *
	w->minus(dw->multiply_num(lambda));
	b->minus(db->multiply_num(lambda));
	*  */
}

void fc_layer::dump()
{
	w->print("w:");
	b->print("b:");
	dw->print("dw:");
	db->print("db:");

	std::cout << "lambda:" << lambda << std::endl;
}

#ifdef __TEST_FC_LAYER__
#undef __TEST_FC_LAYER__

int main()
{
	srand(time(0));
	float w[] = {-10.f, 10.f, 1.0f, 100.f};
	float b[] = {-10.f};
	/* 
	float x[] = {
		0, 1, 
		1, 1, 
		2, 1
	};
	*/
	/* 
	*/
	float x[] = {
		1, 2, 1, 0
			, 6, 5, 4, 6
			, 10, 9, 8, 10 
			, 10, 2, 1, 9
			, 10.1, 2, 1, 8 
			, 6, 100, 4, 5
			, 100, 9, 8, 0 
	};
	/* 
	*/

	int row = sizeof(x) / sizeof(w);
	int mid = sizeof(w) / sizeof(float);

	matrix xarray(x, row, mid);
	matrix standardw(w, mid, 1);
	matrix standardb(b, 1, standardw.get_col());
	standardw.print("expected w :");
	standardb.print("expected b :");

	matrix* yarray = 0;
	matrix::multiply(&yarray, xarray, standardw)->plus_row_row(standardb);

	std::cout << "standard y " << std::endl;
	yarray->print();

	gaussian gau;
	for (int i = 0; i < row; ++i)
		yarray->set_val(i, 0, yarray->get_val(i, 0) + gau.get(0, 0.1f));

	std::cout << "xarray " << std::endl;
	xarray.print();
	std::cout << "~~~~yarray " << std::endl;
	yarray->print();

	/*************************************/
	matrix* input = new matrix(xarray);
	matrix* output = new matrix(*yarray);
	matrix* dx = new matrix(input->get_row(), input->get_col());
	matrix* dy = new matrix(output->get_row(), output->get_col());

	fc_layer fc(input, output, dx, dy, 0.001f);

	int loop = 0;

	std::cout << "loop " << loop << std::endl;
	fc.dump();

	while (loop++ < 5000000) {
		// eval;
		fc.eval();
		// output->print("y: ");

		// dy;
		/*
		if (1)
		{
			std::cout << "loop " << loop << std::endl;
			fc.dump();
			input->print("input:");	
			output->print("output:");
			yarray->print("label y:");
			matrix* ploss = 0;
			matrix::minus(&ploss, *output, *yarray);
			// ploss->print("dyn: ");
			matrix* plosst = 0;
			matrix::transpose(&plosst, *ploss);
			matrix* r = 0;
			matrix::multiply(&r, *plosst, *ploss)->divide_num(yarray->get_row()).print("loss: ");
			delete ploss;
			delete plosst;
			delete r;
		}
		*/

		matrix::minus(&dy, *output, *yarray)->divide_num(yarray->get_row());
		// dy->print("dy: ");

		// bp;
		fc.bp();
	}

	// dump
	std::cout << "loop " << loop << std::endl;
	fc.dump();
}

#endif
