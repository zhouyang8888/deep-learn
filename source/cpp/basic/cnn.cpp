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

#include "cnn.h"
#include "gaussian.h"
#include <algorithm>
#include <iostream>

cnn::cnn(matrix* x, matrix* y, matrix* dx, matrix* dy
		, int xd, int xr, int xc 
		, int nf, int fr, int fc
		, int bz, float lambda) : 
	mx(x), my(y), dmx(dx), dmy(dy) 
	, x(bz), y(bz), f(nf)
	, b(), db()
	, bz(bz)
	, xd(xd), xr(xr + fr - 1), xc(xc + fc - 1) 
	, nf(nf), fd(xd), fr(fr), fc(fc) 
	, yd(nf), yr(xr + fr - 1), yc(xc + fc - 1) 
	, dx(bz), dy(bz), df(nf)
	, lambda(lambda)
	, gd_m_w(0), gd_m_b(0), gd_r_w(0), gd_r_b(0), gd_a_w(0), gd_a_b(0) 
{
	for_each(this->x.begin(), this->x.end(), [&](cube& xe){ xe.resize(this->xd, this->xr, this->xc); });
	for_each(this->y.begin(), this->y.end(), [&](cube& ye){ ye.resize(this->yd, this->yr, this->yc); });
	for_each(this->f.begin(), this->f.end(), [&](cube& fe){ fe.resize(this->fd, this->fr, this->fc); });

	for_each(this->dx.begin(), this->dx.end(), [&](cube& xe){ xe.resize(this->xd, this->xr, this->xc); });
	for_each(this->dy.begin(), this->dy.end(), [&](cube& ye){ ye.resize(this->yd, this->yr, this->yc); });
	for_each(this->df.begin(), this->df.end(), [&](cube& fe){ fe.resize(this->fd, this->fr, this->fc); });

	b.resize(this->yd, this->yr, this->yc);
	db.resize(this->yd, this->yr, this->yc);
	
	gaussian gau;
	for_each(this->f.begin(), this->f.end(), [&](cube& fe) {
			for(auto& frow : fe.v)
			for(auto& fcol : frow)
			for(auto& fdep : fcol)
			fdep = gau.get(0, 0.1f);
			});
	for(auto& bd : b.v)
		for(auto& br : bd)
			for(auto& bc : br) {
				// bc = gau.get(0, 0.f);
				bc = 0.1f;
			}

	// gradient descent.
	// gd_m_w = new momentum(xdim, ydim);
	// gd_m_b = new momentum(1, ydim);
	// gd_r_w = new rmsprop(xdim, ydim);
	// gd_r_b = new rmsprop(1, ydim);
	// gd_a_w = new adam(xdim, ydim);
	// gd_a_b = new adam(1, ydim);
}

cnn::~cnn()
{
	// gd
	// delete gd_m_w;
	// delete gd_m_b;
	// delete gd_r_w;
	// delete gd_r_b;
	// delete gd_a_w;
	// delete gd_a_b;
}

void cnn::eval(const bool train)
{
	to3d(mx, x, xd);

	// rewrite code for efficiency.
	for (int b = 0; b < bz; ++b) {
		cube& out = y[b];
		cube& in = x[b];

		for (int d = 0; d < yd; ++d) {
			cube& f = this->f[d];
			slip(in, f, out[d], this->b[d]);
		} // surface
	} // cube
	
	flatten(my, y, nf);
}

void cnn::slip(cube& x, cube& f, vector<vector<float> >& y, vector<vector<float> >& b)
{
	for (int r = fr / 2; r < yr - fr / 2; ++r) {
		for (int c = fc / 2; c < yc - fc / 2; ++c) {
			int startxr = r - fr / 2;
			int startxc = c - fc / 2;
			float tmp = 0.0f;
			for (int d = 0; d < xd; ++d) {
				for (int fi = 0; fi < fr; ++fi) {
					for (int fj = 0; fj < fc; ++fj) {
						tmp += f[d][fi][fj] * x[d][startxr + fi][startxc + fj];
					}
				}
			}
			y[r][c] = tmp + b[r][c];
		}
	}
}

void cnn::bp()
{
	to3d(dmy, dy, nf);

	difff();
	diffx();
	diffb();

	flatten(dmx, dx, xd);
}

void cnn::difff()
{
	for (int fi = 0; fi < this->nf; ++fi) {
		cube& df = this->df[fi];
		cube& f = this->f[fi];
		for (int fd = 0; fd < this->fd; ++fd) {
			for (int fr = 0; fr < this->fr; ++fr) {
				for (int fc = 0; fc < this->fc; ++fc) {
					float tmp = 0;
					for (int b = 0; b < this->bz; ++b) {
						vector<vector<float> >& x = this->x[b][fd];
						vector<vector<float> >& dy = this->dy[b][fi];
						for (int yr = this->fr / 2; yr < this->yr - this->fr / 2; ++yr) {
							int xr = yr + fr - this->fr / 2;
							for (int yc = this->fc / 2; yc < this->yc - this->fc / 2; ++yc) {
								int xc = yc + fc - this->fc / 2;
								tmp += dy[yr][yc] * x[xr][xc];
							} // dy col
						} // dy row
					} // batch
					df[fd][fr][fc] = tmp;
					f[fd][fr][fc] -= lambda * tmp;
				} // feature col
			} // feature row
		} // feature depth
	} // feature i
}

void cnn::diffb()
{
	for (int yd = 0; yd < this->yd; ++yd) {
		for (int yr = this->fr / 2; yr < this->yr - this->fr / 2; ++yr) {
			for (int yc = this->fc / 2; yc < this->yc - this->fc / 2; ++yc) {
				db[yd][yr][yc] = 0;
				for (int z = 0; z < this->bz; ++z) {
					db[yd][yr][yc] += dy[z][yd][yr][yc];
				}
				b[yd][yr][yc] -= lambda * db[yd][yr][yc];
			}
		}
	}
}

void cnn::diffx()
{
	for (int b = 0; b < bz; ++b) {
		for (int xd = 0; xd < this->xd; ++xd) {
			vector<vector<float> >& dx = this->dx[b][xd];
			for (int xr = this->fr / 2; xr < this->xr - this->fr / 2; ++xr) {
				for (int xc = this->fc / 2; xc < this->xc - this->fc / 2; ++xc) {
					// dx[b][xd][xr][xc]
					float tmp = 0.f;
					for (int fi = 0; fi < this->nf; ++fi) {
						vector<vector<float> >& dy = this->dy[b][fi];
						vector<vector<float> >& f = this->f[fi][xd];
						for (int fr = 0; fr < this->fr; ++fr) {
							int yr = xr - (fr - this->fr / 2);
							for (int fc = 0; fc < this->fc; ++fc) {
								int yc = xc - (fc - this->fc / 2);
								tmp += dy[yr][yc] * f[fr][fc];
							}
						}
					} // feature i
					dx[xr][xc] = tmp;
				} // x col
			} // x row
		} // x depth
	} // batch
}

void print(cube& c)
{
	std::cout << "batch: " << std::endl;
	std::cout << "$$$$$$$$$$depth: " << c.v.size() << std::endl;
	for (auto& d : c.v) {
		std::cout << "surface: " << std::endl;
		for (auto& r : d) {
			std::cout << "row: ";
			for (auto& e : r) {
				std::cout << e << " ";
			}
			std::cout << std::endl;
		}
		std::cout << "end surface. " << std::endl;
	}
	std::cout << "end batch. " << std::endl;
}

void print(vector<cube>& x)
{
	std::cout << "********batch LEN : " << x.size() << std::endl;
	for (auto& c : x) {
		print(c);
	}
}

void cnn::stat(const char* str)
{
	std::cout << str << std::endl;
	// TODO:
	std::cout << "x:========>" << std::endl;
	print(this->x);
	std::cout << "f:========>" << std::endl;
	print(this->f);
	std::cout << "b:========>" << std::endl;
	print(this->b);
	std::cout << "y:========>" << std::endl;
	print(this->y);

	std::cout << std::endl << std::endl;
	std::cout << "dy:========>" << std::endl;
	print(this->dy);
	std::cout << "df:========>" << std::endl;
	print(this->df);
	std::cout << "db:========>" << std::endl;
	print(this->db);
	std::cout << "dx:========>" << std::endl;
	print(this->dx);
}

void cnn::to3d(matrix* mx, vector<cube>& x, int depth)
{
	int mxi = 0;
	for (int b = 0 ; b < bz; ++b) {
		for (int d = 0; d < depth; ++d) {
			for (int r = fr / 2; r < xr - fr / 2; ++r) {
				for_each(x[b][d][r].begin() + fc / 2, x[b][d][r].end() - fc / 2, [&](float& v) {
						v = mx->val[mxi++];
						});
			} // row
		} // surface 
	} // cube 
}

void cnn::flatten(matrix* mx, vector<cube>& x, int depth)
{
	int mxi = 0;
	for (int b = 0 ; b < bz; ++b) {
		for (int d = 0; d < depth; ++d) {
			for (int r = fr / 2; r < xr - fr / 2; ++r) {
				for_each(x[b][d][r].begin() + fc / 2, x[b][d][r].end() - fc / 2, [&](float& v) {
						mx->val[mxi++] = v;
						});
			} // row
		} // surface 
	} // cube 
}

#ifdef __TEST_CNN__
#undef __TEST_CNN__

int main(int argc, char** argv)
{
	srand(time(0));
	float* pf = new float[150];
	for (int i = 0; i < 150; ++i)
		pf[i] = i;

	float* pdf = new float[100];
	for (int i = 0; i < 100; ++i)
		pdf[i] = i;


	matrix* x = new matrix(pf, 2, 75);
	matrix* y = new matrix(2, 50);
	matrix* dx = new matrix(2, 75);
	matrix* dy = new matrix(pdf, 2, 50);

	cnn c(x, y, dx, dy, 3, 5, 5, 2, 3, 3, 2);
	c.eval();
	c.stat("eval");
	c.bp();
	c.stat("bp");
}

#endif
