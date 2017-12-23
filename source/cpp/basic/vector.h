/*
 * =====================================================================================
 *
 *       Filename:  vector.c
 *
 *    Description:  basic ops for vectors.
 *
 *        Version:  1.0
 *        Created:  2017/11/24 08时22分28秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  zhouyang, 
 *   Organization:  
 *
 * =====================================================================================
 */

extern void plus_num(float ret[], float a[],  float b, const int size);


void minus_num(float ret[], float a[],  float b, const int size);


void multiply_num(float ret[], float a[],  float b, const int size);


void divide_num(float ret[], float a[],  float b, const int size);


void plus_tensor(float ret[], float a[],  float b[], const int size);


void minus_tensor(float ret[], float a[],  float b[], const int size);

void dot_product(float* ret, float a[],  float b[], const int size);

void cross_product(float ret[], float a[], float b[], const int sizea, const int sizeb);

void matrix_product(float ret[], float a[], float b[], const int rowa, const int columna, const int columnb);

