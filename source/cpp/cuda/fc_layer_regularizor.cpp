/*
 * =====================================================================================
 *
 *       Filename:  fc_layer_regularizor_regularizor.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/29 20时13分37秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "fc_layer_regularizor.h"

fc_layer_regularizor::fc_layer_regularizor(matrix* x, matrix* y, matrix* dx, matrix* dy, float lambda, float regcoef) : 
	fc_layer(x, y, dx, dy, lambda), regcoef(regcoef)
{
	regdw = new matrix(xdim, ydim);
}

fc_layer_regularizor::~fc_layer_regularizor()
{
	if (regdw) delete regdw;
}

void fc_layer_regularizor::grad()
{
	fc_layer::grad();
	// L2 regularization.
	matrix::multiply_num(&regdw, *w, regcoef);
	dw->plus(*regdw);
}
