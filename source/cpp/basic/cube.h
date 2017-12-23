/*
 * =====================================================================================
 *
 *       Filename:  image.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/14 12时53分34秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __cube__
#define __cube__

#include <algorithm>
#include <vector>

using namespace std;

class cube {
	public:
		vector<vector<vector<float> > > v;
		int r;
		int c;
		int d;

	public:
		inline cube () : v(), r(0), c(0), d(0) {}
		
		inline vector<vector<float> >& operator[](int i) { return v[i]; }
		inline void resize(int r, int c, int d) {
			this->r = r;
			this->c = c;
			this->d = d;
			v.resize(r);
			for_each(v.begin(), v.end(), [&](vector<vector<float> >& ve) {
					ve.resize(c);
					for_each(ve.begin(), ve.end(), [&](vector<float>& vf) {
						vf.resize(d, 0.0f);
						});
					});
		}
		inline int size() {
			return v.size();
		}
};

#endif
