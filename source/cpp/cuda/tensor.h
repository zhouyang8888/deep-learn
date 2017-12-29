/*
 * =====================================================================================
 *
 *       Filename:  tensor.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/24 11时15分14秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __tensor__
#define __tensor__
class tensor {
	float* val;
	int size;

	public:
		tensor(int size);
		tensor(float* val, int size);
		tensor(const tensor& other);
		virtual ~tensor();

		inline const float* get_val() const {
			return val;
		}
		inline int get_size() const {
			return size;
		}

		void plus_num(const float b);
		void plus_num(tensor& ret, const float b) const;

		void minus_num(const float b);
		void minus_num(tensor& ret, const float b) const;

		void multiply_num(const float b);
		void multiply_num(tensor& ret, const float b) const;

		void divide_num(const float b);
		void divide_num(tensor& ret, const float b) const;

		void plus_tensor(const tensor& b);
		void plus_tensor(tensor& ret, const tensor& b) const;

		void minus_tensor(const tensor& b);
		void minus_tensor(tensor& ret, const tensor& b) const;

		float dot_product(const tensor& b) const;
};

#endif
