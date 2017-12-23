/*
 * =====================================================================================
 *
 *       Filename:  avgpooling.h
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

#include "avgpooling.h"
#include <cassert>
#include <iostream>

avgpooling::avgpooling(matrix* x, matrix* y, matrix* dx, matrix* dy
		, int xd, int xr, int xc, int rstep, int cstep)
	: x(x), y(y), dx(dx), dy(dy)
	  , xd(xd), xr(xr), xc(xc), rstep(rstep), cstep(cstep)
	  , ywidth(xc / cstep), yheight(xr / rstep)
	  , xarea(xr * xc), yarea(xr / rstep * xc / cstep)
{
	assert(y->get_row() == x->get_row());
	assert(x->get_col() == xd * xr * xc);
	assert(y->get_col() == x->get_col() / rstep / cstep);
}

avgpooling::~avgpooling()
{
}

void avgpooling::eval()
{
	for (int bi = 0; bi < x->get_row(); ++bi) {
		int sample = bi * x->get_col();
		int label = bi * y->get_col();
		for (int di = 0; di < xd; ++di) {
			int sampledi = sample + di * xarea;
			int labeldi = label + di * yarea;
			for (int yri = 0; yri < yheight; ++yri) {
				for (int yci = 0; yci < ywidth; ++yci) {
					int cury = labeldi + yri * ywidth + yci;
					y->val[cury] = 0.0f;
					for (int xri = yri * rstep; xri < yri * rstep + rstep; ++xri) {
						for (int xci = yci * cstep; xci < yci * cstep + cstep; ++xci) {
							float tmp = x->val[sampledi + xri * xc + xci];
							y->val[cury] += tmp;
						}
					}
					y->val[cury] /= (rstep * cstep);
				}
			}
		}
	}
}

void avgpooling::bp()
{
	for (int bi = 0; bi < x->get_row(); ++bi) {
		int sample = bi * x->get_col();
		int label = bi * y->get_col();
		for (int di = 0; di < xd; ++di) {
			int sampledi = sample + di * xarea;
			int labeldi = label + di * yarea;
			for (int yri = 0; yri < yheight; ++yri) {
				for (int yci = 0; yci < ywidth; ++yci) {
					int cury = labeldi + yri * ywidth + yci;
					for (int xri = yri * rstep; xri < yri * rstep + rstep; ++xri) {
						for (int xci = yci * cstep; xci < yci * cstep + cstep; ++xci) {
							dx->val[sampledi + xri * xc + xci] = dy->val[cury] / (rstep * cstep);
						}
					}
				}
			}
		}
	}
}

void avgpooling::stat(const char* str)
{
	if (str) std::cout << str << std::endl;
	x->print("x:");
	y->print("y:");
	dy->print("dy:");
	dx->print("dx:");
}

#ifdef __TEST_AVGPOOLING__
#undef __TEST_AVGPOOLING__

int main(int argc, char** argv)
{
	float* pf = new float[48];
	for (int i = 0; i < 48; ++i)
		pf[i] = i;

	float* pdf = new float[48];
	for (int i = 0; i < 48; ++i)
		pdf[i] = i;

	matrix* x = new matrix(pf, 2, 24);
	matrix* y = new matrix(2, 6);
	matrix* dx = new matrix(2, 24);
	matrix* dy = new matrix(pdf, 2, 6);

	avgpooling mp(x, y, dx, dy, 3, 4, 2, 2, 2);
	mp.eval();
	mp.bp();
	mp.stat("avgpooling:");

	return 0;
}

#endif
