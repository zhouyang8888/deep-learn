/*
 * =====================================================================================
 *
 *       Filename:  test.cpp
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/11/28 12时38分07秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "fc_layer_regularizor.h"
#include "sigmoid_layer.h"
#include "relu_layer.h"
#include "tanh_layer.h"
#include "prelu_layer.h"
#include "softplus_layer.h"
#include "matrix0.h"
#include "gaussian.h"
#include "ele_op.h"
#include "batch_normalize.h"
#include "net.h"
#include <iostream>
#include <cmath>
#include <vector>
#include <queue>
#include <cassert>
#include <algorithm>

// #define SIZE 327680
// #define SIZE 163840
#define SIZE 81920
// #define SIZE 40960
// #define SIZE 20
// #define BATCH_SIZE 1024
// #define BATCH_SIZE 512
#define BATCH_SIZE 256
// #define BATCH_SIZE 128 
// #define BATCH_SIZE 64 
// #define BATCH_SIZE 16
#define PI 3.1415926
#define LOOP 50000
#define LAMBDA 0.05f

matrix** return_spirals(const int train_size, const int test_size)
{
	const int double_train_size = train_size;

	float **x = new float*[2];
	float **y = new float*[2];

	const int TRAIN = 0;
	const int TEST  = 1;
	x[TRAIN] = new float[train_size << 1];
	y[TRAIN] = new float[train_size];
	x[TEST] = new float[test_size << 1];
	y[TEST] = new float[test_size];

	gaussian gau;

	for (int index = 0; index < train_size; ++index) {
		float theta = ((rand() % 2000) * 2 * PI) / 500; 
		int pi_count = rand() % 2;
		float r = theta + pi_count * PI + gau.get(0, 0.1f);
		x[TRAIN][index * 2] = r * cos(theta);
		x[TRAIN][index * 2 + 1] = r * sin(theta);
		y[TRAIN][index] = pi_count;

		++index;
	}

	for (int index = 0; index < test_size; ++index) {
		float theta = ((rand() % 2000) * 2 * PI) / 500; 
		int pi_count = rand() % 2;
		float r = theta + pi_count * PI + gau.get(0, 0.1f);
		x[TEST][index * 2] = r * cos(theta);
		x[TEST][index * 2 + 1] = r * sin(theta);
		y[TEST][index] = pi_count;

		++index;
	}

	matrix* input_t = new matrix(x[TRAIN], train_size, 2);
	matrix* output_t = new matrix(y[TRAIN], train_size, 1);
	matrix* input_x = new matrix(x[TEST], test_size, 2);
	matrix* output_x = new matrix(y[TEST], test_size, 1);
	output_t->print();
	exit(0);

	for (int i = 0; i < 2; ++i) {
		delete[] x[i];
		delete[] y[i];
	}
	delete[] x;
	delete[] y;

	matrix** ret = new matrix*[4];
	ret[0] = input_t;
	ret[1] = output_t;
	ret[2] = input_x;
	ret[3] = output_x;

	return ret;
}

matrix** return_circles(const int train_size, const int test_size)
{
	const int double_train_size = train_size;

	float **x = new float*[2];
	float **y = new float*[2];

	const int TRAIN = 0;
	const int TEST  = 1;
	x[TRAIN] = new float[train_size << 1];
	y[TRAIN] = new float[train_size];
	x[TEST] = new float[test_size << 1];
	y[TEST] = new float[test_size];

	gaussian gau;

	float v = 0.1;
	float omega = 0.1;
	float v1 = 0.2;
	float omega2 = 0.1;

	for (int index = 0; index < train_size; ++index) {
			int t = rand() % 100; 
			float theta = 2 * PI * t / 100 + gau.get(0, 0.1f);

			float stdr = (float)(rand() % 8) / 2.0f + 1;
			float r = gau.get(stdr, 0.2f);
			
			x[TRAIN][index * 2] = r * cos(theta);
			x[TRAIN][index * 2 + 1] = r * sin(theta);
			y[TRAIN][index] = ((int)(stdr * 2)) % 2;
	}

	for (int index = 0; index < test_size; ++index) {
			int t = rand() % 100; 
			float theta = 2 * PI * t / 100 + gau.get(0, 0.1f);

			float stdr = (float)(rand() % 8) / 2.0f + 1;
			float r = gau.get(stdr, 0.2f);
			
			x[TEST][index * 2] = r * cos(theta);
			x[TEST][index * 2 + 1] = r * sin(theta);
			y[TEST][index] = ((int)(stdr * 2)) % 2;
	}

	matrix* input_t = new matrix(x[TRAIN], train_size, 2);
	matrix* output_t = new matrix(y[TRAIN], train_size, 1);
	matrix* input_x = new matrix(x[TEST], test_size, 2);
	matrix* output_x = new matrix(y[TEST], test_size, 1);
	for (int i = 0; i < 2; ++i) {
		delete[] x[i];
		delete[] y[i];
	}
	delete[] x;
	delete[] y;

	matrix** ret = new matrix*[4];
	ret[0] = input_t;
	ret[1] = output_t;
	ret[2] = input_x;
	ret[3] = output_x;

	return ret;
}

void test_sigmoid()
{
	int width[] = {
		100 // 0 : fc0
		, 100  // 1 : non-linear 
		, 50 //  2 : fc0
		, 50  // 3 : non-linear 
		, 20 //  4 : fc0
		, 20  // 5 : non-linear 
		, 10 //  6 : fc0
		, 10  // 7 : non-linear 
		, 1  // 8 : fc1
		, 1  // 9 : non-linear 
	};
	// matrix** data_set = return_spirals(SIZE, SIZE/10);
	matrix** data_set = return_circles(SIZE, SIZE/10);
	matrix** train_set = new matrix*[2];
	train_set[0] = data_set[0];
	train_set[1] = data_set[1];
	matrix** test_set = new matrix*[2];
	test_set[0] = data_set[2];
	test_set[1] = data_set[3];

	delete[] data_set;

	std::cout << "train set" << std::endl;
	// train_set[0]->print();
	std::cout << "test  set" << std::endl;
	// test_set[0]->print();

	class subnet : public net {
		public:
			subnet(int bs, int f) : net(bs, f) {

			}

			float loss() {
				return loss_square();
			};

			float loss_square() {
				matrix* output = net::x[net::x.size() - 1];
				matrix* expected = net::expected;

				float sum = 0;
				for (int i = 0; i < output->get_row(); ++i) {
					float code = - expected->get_val(0, i) + output->get_val(0, i);

					sum += code * code;
				}

				sum /= net::batch_size;
			
				return sum;
			};

			float loss_prob() {
				matrix* output = net::x[net::x.size() - 1];
				matrix* expected = net::expected;

				float sum = 0;
				for (int i = 0; i < output->get_row(); ++i) {
					float code = expected->get_val(0, i) > 0.5f ? 
						-log(output->get_val(0, i)) : 
						-log(1.0f - output->get_val(0, i));

					sum += code;
				}

				sum /= net::batch_size;
			
				return sum;
			};

			void dloss() {
				dloss_square();
			};

			void dloss_square() {
				matrix* output = x[x.size() - 1];
				matrix* expected = net::expected;
				matrix* dloss_dy = dx[dx.size() - 1];

				for (int i = 0; i < output->get_row(); ++i) {
					float code = - expected->get_val(0, i) + output->get_val(0, i);

					dloss_dy->set_val(0, i, code / batch_size);
				}
			};

			void dloss_prob() {
				matrix* output = x[x.size() - 1];
				matrix* expected = net::expected;
				matrix* dloss_dy = dx[dx.size() - 1];

				for (int i = 0; i < output->get_row(); ++i) {
					float code = expected->get_val(0, i) > 0.5f ? 
						(-1.0 / output->get_val(0, i)) : 1.0f / (1.0f - output->get_val(0, i));

					dloss_dy->set_val(0, i, code / batch_size);
				}
			};

			float accuracy() {
				matrix* output = net::x[net::x.size() - 1];
				matrix* expected = net::expected;

				float sum = 0;
				for (int i = 0; i < output->get_row(); ++i) {
					float diff = output->get_val(0, i) - expected->get_val(0, i);
					if (diff > -0.5f && diff < 0.5f) sum += 1;
				}

				sum /= net::batch_size;
			
				return sum;
			};
	};
	subnet n(BATCH_SIZE, train_set[0]->get_col());
	subnet n2(BATCH_SIZE, train_set[0]->get_col());

	std::vector<int> bns;
	std::vector<int> fcs;

	n.init();
	n2.init();

	n.add_fc(width[0], LAMBDA);
	n2.add_fc(width[0], LAMBDA);
	fcs.push_back(0);

	n.add_bn(width[0], LAMBDA);
	bns.push_back(1);

	n.add_sigmoid(width[1]);
	n2.add_sigmoid(width[1]);
	// n.add_relu(width[1]);
	// n.add_tanh(width[1]);
	// n.add_prelu(width[1], LAMBDA);
	// n.add_softplus(width[1]);
	// n.add_bn(width[1], LAMBDA);

	n.add_fc(width[2], LAMBDA);
	n2.add_fc(width[2], LAMBDA);
	fcs.push_back(3);

	n.add_bn(width[2], LAMBDA);
	bns.push_back(4);
	// n2.add_bn(width[2], LAMBDA);
	n.add_sigmoid(width[3]);
	n2.add_sigmoid(width[3]);
	// n.add_relu(width[3]);
	// n.add_tanh(width[3]);
	// n.add_prelu(width[3], LAMBDA);
	// n.add_softplus(width[3]);
	// n.add_bn(width[3], LAMBDA);

	//
	n.add_fc(width[4], LAMBDA);
	n2.add_fc(width[4], LAMBDA);
	fcs.push_back(6);

	n.add_bn(width[4], LAMBDA);
	bns.push_back(7);
	// n2.add_bn(width[4], LAMBDA);
	n.add_sigmoid(width[5]);
	n2.add_sigmoid(width[5]);
	// n.add_relu(width[5]);
	// n.add_tanh(width[5]);
	// n.add_prelu(width[5], LAMBDA);
	// n.add_softplus(width[5]);
	// n.add_bn(width[5], LAMBDA);

	n.add_fc(width[6], LAMBDA);
	n2.add_fc(width[6], LAMBDA);
	fcs.push_back(9);

	n.add_bn(width[6], LAMBDA);
	bns.push_back(10);
	// n2.add_bn(width[6], LAMBDA);
	
	n.add_sigmoid(width[7]);
	n2.add_sigmoid(width[7]);
	// n.add_relu(width[7]);
	// n.add_tanh(width[7]);
	// n.add_prelu(width[7], LAMBDA);
	// n.add_softplus(width[7]);
	// n.add_bn(width[7], LAMBDA);

	n.add_fc(width[8], LAMBDA);
	n2.add_fc(width[8], LAMBDA);
	fcs.push_back(12);

	n.add_bn(width[8], LAMBDA);
	bns.push_back(13);
	// n2.add_bn(width[8], LAMBDA);
	n.add_sigmoid(width[9]);
	n2.add_sigmoid(width[9]);
	//
	// n.add_relu(width[9]);
	// n.add_tanh(width[9]);
	// n.add_prelu(width[9], LAMBDA);
	// n.add_softplus(width[9]);
	// n.add_bn(width[9], LAMBDA);

	n.finish();
	n2.finish();

	// collapse from front to end.
	for (int i = 0; i < bns.size(); ++i) {
		bns[i] -= i;
		fcs[i] -= i;
	}

	std::cout << "xxxx" << std::endl;
	for (int i = 0; i < fcs.size(); ++i)
		std::cout << fcs[i] << ", " << bns[i] << std::endl;

	int loop = 0;

	bool collapsing = false;

	std::vector<int> bn_sn;
	std::vector<int> fc_sn;

	// int lastupdate = 0;
	// int updatecount = 0;
	while (loop++ < LOOP) {

		bool dump = (loop % 100 == 1);
		matrix** data = train_set;
		if (dump) {
			std::cout << "dump:" << std::endl;
			data = test_set;
		}

		int xtotalsize = data[0]->get_row() * data[0]->get_col();
		int batch_xdata_len = BATCH_SIZE * data[0]->get_col();
		int batch_ydata_len = BATCH_SIZE * data[1]->get_col();

		int startxpos = 0;
		int startypos = 0;
		
		float loss = 0.0f;
		float loss2 = 0.0f;
		float accuracy = 0;
		float accuracy2 = 0;
		int batch_count = 0;
		do {
			n.fill_in_x(data[0]->get_val() + startxpos, batch_xdata_len);
			n2.fill_in_x(data[0]->get_val() + startxpos, batch_xdata_len);
			n.fill_in_y(data[1]->get_val() + startypos, batch_ydata_len);
			n2.fill_in_y(data[1]->get_val() + startypos, batch_ydata_len);

			n.eval();
			n2.eval();

			if (collapsing) {
				n.collect(bn_sn);
			}

			loss += n.loss();
			loss2 += n2.loss();
			accuracy += n.accuracy();
			accuracy2 += n2.accuracy();

			if (dump) {
				/* 
				const matrix* batch_in = n.get_input();
				const matrix* n_out = n.get_output();
				const matrix* n2_out = n2.get_output();
				for (int i = 0; i < batch_in->get_row(); ++i) {
					for (int j = 0; j < batch_in->get_col(); ++j) {
						std::cout << batch_in->get_val(i, j) << " ";
					}
					std::cout << n_out->get_val(i, 0) << " ";
					std::cout << n2_out->get_val(i, 0) << " ";
					std::cout << std::endl;
				} */
			} else {
				n.dloss();
				n2.dloss();

				n.bp();
				n2.bp();
			}

			startxpos += batch_xdata_len;
			startypos += batch_ydata_len;
			++batch_count;
		}while(startxpos < xtotalsize);

		const char* type = dump ? "test] : " : "train] : ";
		if (dump) {
			std::cout << "end dump batch" << std::endl;
		} 

		std::cout << type << loop << ", net     bn, loss: " << loss / batch_count << std::endl;
		std::cout << type << loop << ", net     bn, accuracy: " << accuracy / batch_count * 100 << "%" << std::endl;
		std::cout << type << loop << ", net[NO] bn, loss: " << loss2 / batch_count << std::endl;
		std::cout << type << loop << ", net[NO] bn, accuracy: " << accuracy2 / batch_count * 100 << "%" << std::endl;

		/* *
		if (updatecount < 3 && accuracy > 0.95f * batch_count && lastupdate < loop - 200) {
			n.update_lambda(0.5f);
			lastupdate = loop;
			++updatecount;
		}
		* */
		if (!collapsing && !fcs.empty() && accuracy / batch_count > 0.95f) {
			fc_sn.push_back(*fcs.begin());
			fcs.erase(fcs.begin());
			bn_sn.push_back(*bns.begin());
			bns.erase(bns.begin());
			std::cout << "Begin collect : fc[" << fc_sn[0] << "] bn[" << bn_sn[0] << "]" << std::endl;
			n.begin_collect(bn_sn);
			collapsing = true;
		} else if (collapsing) {
			n.end_collect(bn_sn);
			std::cout << "End collect" << std::endl;
			n.collapse_bn_to_fc(bn_sn, fc_sn);
			n.dump_output();
			std::cout << "collapse ended." << std::endl;
			collapsing = false;
			fc_sn.clear();
			bn_sn.clear();
		} else {
			;
		}
	}; // end loop
}

int main(int argc, char** argv)
{
	test_sigmoid();
}
