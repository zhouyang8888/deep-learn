/*
 * =====================================================================================
 *
 *       Filename:  net_layer.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/27 08时33分23秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __net_layer__
#define __net_layer__

#include "matrix0.h"

class net_layer {
	public:
		virtual void eval() = 0;
		virtual void bp() = 0;
		virtual void stat(const char* str) = 0;
		inline net_layer() {
		}
		inline virtual ~net_layer() {
		}
		inline virtual void set_lambda(float lambda) {}
		inline virtual float get_lambda() { return 0.f; }
		inline virtual void decay_lambda(float v) { set_lambda(v * get_lambda()); }

		virtual matrix*& get_y() = 0;
		virtual matrix*& get_dy() = 0;
		virtual matrix*& get_x() = 0; 
		virtual matrix*& get_dx() = 0; 
};

#endif

