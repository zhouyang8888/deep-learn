/*
 * =====================================================================================
 *
 *       Filename:  gaussian.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/25 08时16分05秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __gaussian__
#define __gaussian__

// #include <atomic>

class gaussian {
	private:
		bool generate;
		double z1;
		double z0;

	public:
		gaussian();
		virtual ~gaussian();
		double get();
		double get(double mean, double stdvar);

	private:
		// TODO: running : sync op, TO BE IMPLEMENTED.
		// atomic_bool running;
		static const double epsilon;
		static const double two_pi;
};

#endif
