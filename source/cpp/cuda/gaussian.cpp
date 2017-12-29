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

#include "gaussian.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <time.h>
#include <limits>

const double gaussian::epsilon = std::numeric_limits<double>::min();
const double gaussian::two_pi = 2 * 3.14159265358979323846;

gaussian::gaussian() 
{
	generate = 1;
	z0 = z1 = 0;
}

gaussian::~gaussian() 
{
}

double gaussian::get() 
{
	if (generate) {
		double u1,  u2;
		do {
			u1 = rand() * (1.0 / RAND_MAX);
			u2 = rand() * (1.0 / RAND_MAX);
		} while (u1 <= epsilon || u2 <= epsilon);

		z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
		z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);

		generate = false;
		return z0;
	} else {
		generate = true;
		return z1;
	}
}

double gaussian::get(double mean, double stdvar)
{
	return get() * stdvar + mean;
}

#ifdef __TEST_GAUSSIAN__
#undef __TEST_GAUSSIAN__
int main()
{
	srand(time(0));
#define TOTAL 1000000
#define INTERVAL_COUNT 20 
#define HEIGHT 10 

	gaussian gau;
	float val[TOTAL];
	float min = 0.0f, max = 0.0f;
	for (int i = 0; i < TOTAL; ++i) {
		val[i] = (float) gau.get();
		// std::cout << val[i] << std::endl;

		if (min > val[i]) min = val[i];
		if (max < val[i]) max = val[i];
	}

	float width = (max - min) / INTERVAL_COUNT;
	int count[INTERVAL_COUNT + 1];
	memset(count, 0, sizeof(int) * 11);
	for (int i = 0; i < TOTAL; ++i) {
		++count[(int)((val[i] - min) / width)];
	}
	int cmax = 0;
	for (int i = 0; i < INTERVAL_COUNT + 1; ++i) {
		cmax = cmax < count[i] ? count[i] : cmax;
	}

	for (int i = 0; i < INTERVAL_COUNT + 1; ++i)
		count[i] = count[i] * HEIGHT / cmax;

	int threshold = HEIGHT;
	while (threshold >= 0) {
		for (int i = 0; i < INTERVAL_COUNT + 1; ++i) {
			if (count[i] >= threshold)
				std::cout << 'H';
			else 
				std::cout << ' ';
		}
		std::cout << std::endl;
		--threshold;
	}
}
#endif
