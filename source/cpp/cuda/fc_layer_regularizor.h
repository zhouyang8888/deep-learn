/*
 * =====================================================================================
 *
 *       Filename:  fc_layer_regularizor.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/29 20时09分00秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __FC_LAYER_REG__
#define __FC_LAYER_REG__

#include "fc_layer.h"

class fc_layer_regularizor : public fc_layer {

	private:
		float regcoef;
		matrix* regdw;
	public:
		fc_layer_regularizor(matrix* x, matrix* y, matrix* dx, matrix* dy, float lambda = 0.01f, float regcoef = 0.1f);
		virtual ~fc_layer_regularizor();
	private:
		void grad();
};


#endif
