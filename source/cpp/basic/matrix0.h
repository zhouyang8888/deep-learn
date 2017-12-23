/*
 * =====================================================================================
 *
 *       Filename:  matrix.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/24 12时04分37秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#ifndef __matrix__
#define __matrix__

#include "tensor.h"

class matrix {
	friend class ele_op;
	friend class batch_normalize;

	public:
		float* val;
	private:
		int row;
		int col;
	private:
		// inline float* __get_val() { return val; }

	public:
		matrix(int row, int col);

		matrix(const float* val, int row, int col);

		matrix(const tensor** tensors, int row, int col);

		matrix(const matrix& other);

		virtual ~matrix();

		inline const float get_val(int i, int j) const { return val[i * col + j]; }
		inline void set_val(int i, int j, float v) { val[i * col + j] = v; }

		inline const int get_row() const { return row; }
		inline const int get_col() const { return col; }
		inline const float* get_val() const { return val; }

		matrix& transpose();
		// matrix& multiply(const matrix& other);
		matrix& plus(const matrix& mat);
		matrix& plus_row_row(const matrix& row);
		matrix& plus_col_col(const matrix& col);
		matrix& minus(const matrix& mat);

		matrix& multiply_num(float coef);
		matrix& divide_num(float coef);

		static matrix* multiply_num(matrix** ret, const matrix& mat, float coef);
		static matrix* divide_num(matrix** ret, const matrix& mat, float coef);
		static matrix* transpose(matrix** ret, const matrix& mat);
		static matrix* multiply(matrix** ret, const matrix& mat0,  const matrix& mat1);
		static matrix* plus(matrix** ret, const matrix& mat0,  const matrix& mat1);
		static matrix* minus(matrix** ret, const matrix& mat0,  const matrix& mat1);
		static matrix* sum_rows(matrix** ret, const matrix& mat);
		static matrix* sum_cols(matrix** ret, const matrix& mat);

		matrix& copy_from(const matrix& other);
		matrix& copy_by_row(const matrix& mat, int startrow, int count);
		matrix& copy_from_array(const float* arr, int len);

		void print(const char* line = 0) const;
};

#endif
