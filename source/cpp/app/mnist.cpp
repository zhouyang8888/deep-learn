/*
 * =====================================================================================
 *
 *       Filename:  mnist.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  2017/12/11 09时28分52秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include "mnist.h"
#include "net.h"
#include "matrix0.h"
#include "color_macro.h"
#include <cstdio>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

#define LAMBDA 0.05
#define LOOP 100

mnist::mnist(const char* path_prefix) : train_image_file(path_prefix), train_label_file(path_prefix) 
										, test_image_file(path_prefix), test_label_file(path_prefix)
{
	train_image_file += "train-images-idx3-ubyte";
	train_label_file += "train-labels-idx1-ubyte";
	test_image_file += "t10k-images-idx3-ubyte";
	test_label_file += "t10k-labels-idx1-ubyte";
}

void mnist::construct_net()
{
	int width[] = {
			1000       // fc  0
			, 500       // fc  0
			, 200       // fc  2
			, 50       // fc  2
			, 20       // fc  2
			, 10          // fc  3
	};

	class net_impl: public net {
		public:
			net_impl(int bs) : net(bs, 28 * 28) {

			}

			float loss() {
				return loss_square();
			};

			float loss_square() {
				matrix* output = net::x[net::x.size() - 1];
				matrix* expected = net::expected;

				float sum = 0;
				for (int i = 0; i < output->get_row(); ++i) {
					int targetj = 0;
					for (int j = 0; j < expected->get_col(); ++j) {
						if (1 == expected->get_val(i, j)) {
							targetj = j;
							break;
						}
					}
					float targetv = output->get_val(i, targetj);

					for (int j = 0; j < output->get_col(); ++j) {
						if (j == targetj) continue;
						float curv = output->get_val(i, j);
						if (targetv - 0.1f < curv) {
							sum += (curv - targetv) * (curv - targetv);
						}
					}
				}

				sum /= net::batch_size;
			
				return sum;
			};

			float loss_prob() {
				// TODO:
				matrix* output = net::x[net::x.size() - 1];
				matrix* expected = net::expected;

				float sum = 0;
				for (int i = 0; i < output->get_row(); ++i) {
					int targetj = 0;
					for (int j = 0; j < expected->get_col(); ++j) {
						if (1 == expected->get_val(i, j)) {
							targetj = j;
							break;
						}
					}
					float targetv = output->get_val(i, targetj);

					for (int j = 0; j < output->get_col(); ++j) {
						if (j == targetj) continue;
						float curv = output->get_val(i, j);
						if (targetv - 0.1f < curv) {
							sum += (curv - targetv) * (curv - targetv);
						}
					}
				}

				return sum;
			};

			void dloss() {
				dloss_square();
			};

			void dloss_square() {
				matrix* output = x[x.size() - 1];
				matrix* expected = net::expected;
				matrix* dloss_dy = dx[dx.size() - 1];

				// std::cout << std::endl;
				for (int i = 0; i < output->get_row(); ++i) {
					int targetj = 0;
					for (int j = 0; j < expected->get_col(); ++j) {
						if (1 == expected->get_val(i, j)) {
							targetj = j;
							break;
						}
					}
					float targetv = output->get_val(i, targetj);
					
					float sum = 0;
					for (int j = 0; j < output->get_col(); ++j) {
						/* * *
						{
							if (j != targetj) {
								if (output->get_val(i, j) >= targetv)
									std::cout << color_macro::YELLOW << output->get_val(i, j) << " ";
								else if (output->get_val(i, j) >= targetv - 0.1f)
									std::cout << color_macro::MAGENTA << output->get_val(i, j) << " ";
								else
									std::cout << color_macro::RESET << output->get_val(i, j) << " ";
							}
							else 
								std::cout << color_macro::RED << output->get_val(i, j) << " ";
						}
						* * */
						if (j == targetj) continue;
						float curv = output->get_val(i, j);
						if (targetv - 0.1f < curv) {
							dloss_dy->set_val(i, j, (curv - targetv + 0.1f) * 10 / net::batch_size);
							sum += (targetv - curv - 0.1f) * 10;
						} else {
							dloss_dy->set_val(i, j, 0);
						}
					} // end for image 
					// std::cout << std::endl;
					dloss_dy->set_val(i, targetj, sum / net::batch_size);
				} // end for batch
				// std::cout << std::endl;
			};

			void dloss_prob() {
				// TODO:
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
					int maxj = 0;
					float maxv = output->get_val(i, 0);
					for (int j = 1; j < output->get_col(); ++j) {
						float curv = output->get_val(i, j);
						if (maxv < curv) {
							maxv = curv;
							maxj = j;
						}
					}
					int targetj = 0;
					for (int j = 0; j < expected->get_col(); ++j) {
						if (1 == expected->get_val(i, j)) {
							targetj = j;
							break;
						}
					}
					if (maxj == targetj) sum += 1;
				}

				sum /= net::batch_size;
			
				return sum;
			};
	};

	mnist_net = new net_impl(batch_size);
	mnist_net->init();
	int level_num = sizeof(width) / sizeof(int);
	for (int i = 0; i < level_num - 1; ++i) {
		mnist_net->add_fc(width[i], LAMBDA);
		// mnist_net->add_bn(width[i], LAMBDA);
		// mnist_net->add_sigmoid(width[i]);
		mnist_net->add_relu(width[i]);
	}
	mnist_net->add_fc(width[level_num - 1], LAMBDA);
	mnist_net->add_bn(width[level_num - 1], LAMBDA);
	mnist_net->add_sigmoid(width[level_num - 1]);

	mnist_net->finish();

	int loop = 0; 
	float accu = 0;
	float loss = 0;
	float vaccu = 0;
	float vloss = 0;
	float taccu = 0;
	float tloss = 0;
	while (loop++ < LOOP) {
		std::cout << "Loop " << loop << std::endl;
		std::cout << "Train set:" << std::endl;
		mnist_net->epoch(accu, loss, train_image, train_label);
		std::cout << "acc: " << accu * 100 << "%, " << "loss: " << loss << "." << std::endl;

		if (0 == loop % 5) {
			float vcaccu = 0;
			float vcloss = 0;
			std::cout << "Valid set:" << std::endl;
			mnist_net->eval(vcaccu, vcloss, valid_image, valid_label);
			std::cout << "acc: " << vcaccu * 100 << "%, " 
				<< "loss: " << vcloss << "." << std::endl;
			if ((vcaccu - vaccu) < (1 - vaccu) / 10) {
				std::cout << "No obvious promotion, exit" << std::endl;
				break;
			}
		}
	};

	std::cout << "Test set:" << std::endl;
	mnist_net->eval(taccu, tloss, test_image, test_label);
	std::cout << "acc: " << taccu * 100 << "%, " 
		<< "loss: " << tloss << "." << std::endl;
}

// shrink image from 28 * 28 to 28 * 14 
void mnist::shrink_image(unsigned char* orig, float* vec)
{
	for (int line_num = 0; line_num < 14; ++line_num) {
		unsigned char* f = orig + line_num * 2 * 28;
		unsigned char* s = f + 28;
		float* pv = vec + line_num * 14;

		for(int i = 0; i < 14; ++i) {
			pv[i] = std::max(
					std::max((unsigned int)(f[0]), (unsigned int)(f[1]))
					, std::max((unsigned int)(s[0]), (unsigned int)(s[1])));
			f += 2;
			s += 2;
		}
	}

	/* * *
	for (int i = 0, j = 0; i < 28 * 28; ++i, ++j) {
		if (j % 28 == 0) std::cout << std::endl;
		if (orig[i] != 0)
			std::cout << color_macro::RED << (unsigned int)orig[i] << " ";
		else
			std::cout << color_macro::RESET << (unsigned int)orig[i] << " ";
	}
	for (int i = 0, j = 0; i < 14 * 14; ++i, ++j) {
		if (j % 14 == 0) std::cout << std::endl;
		if (vec[i] != 0)
			std::cout << color_macro::RED << vec[i] << " ";
		else
			std::cout << color_macro::RESET << vec[i] << " ";
	}
	* * */
}

void mnist::load_data()
{
	unsigned char* train_image_data = read_image_file(train_image_file, 60000, 0x00000803);
	unsigned char* train_label_data = read_label_file(train_label_file, 60000, 0x00000801);

	unsigned char* test_image_data = read_image_file(test_image_file, 10000, 0x00000803);
	unsigned char* test_label_data = read_label_file(test_label_file, 10000, 0x00000801);

    int train_size = 55000;
    int valid_size = 60000 - train_size;
    int test_size = 10000;

	train_image = new matrix(train_size, 28 * 28);
	train_label= new matrix(train_size, 10);

	valid_image = new matrix(valid_size, 28 * 28);
	valid_label = new matrix(valid_size, 10);

	test_image = new matrix(test_size, 28 * 28);
	test_label = new matrix(test_size, 10);

	// train
	for (int i = 0; i < train_size * 28 * 28; ++i) {
		train_image->val[i] = (unsigned int)train_image_data[i];
	}
	for (int i = 0; i < train_size; ++i) {
		train_label->val[i * 10 + (unsigned int)train_label_data[i]] = 1;
	}

	// validation
	for (int i = 0; i < valid_size * 28 * 28; ++i) {
		valid_image->val[i] = (unsigned int)train_image_data[i + train_size * 28 * 28];
	}
	for (int i = 0; i < valid_size; ++i) {
		valid_label->val[i * 10 + (unsigned int)train_label_data[i + train_size]] = 1;
	}

	// test
	for (int i = 0; i < test_size * 28 * 28; ++i) {
		test_image->val[i] = (unsigned int)test_image_data[i];
	}
	for (int i = 0; i < test_size; ++i) {
		test_label->val[i * 10 + (unsigned int)test_label_data[i]] = 1;
	}

	delete[] train_image_data;
	delete[] train_label_data;
	delete[] test_image_data;
	delete[] test_label_data;
}

unsigned char* mnist::read_label_file(const string& file, int image_count, int file_magic_num)
{
	int magic_num = 0;
	int image_num = 0;

	FILE* f = fopen(file.c_str(), "r");
	if (0 == f) {
		std::cerr << "Open \"" << file << "\" error." << std::endl;
		exit(-1);
	}

	size_t n = fread(&magic_num, sizeof(int), 1, f);
	int_reverse(magic_num);
	assert(n == 1 && magic_num == file_magic_num);

	n = fread(&image_num, sizeof(int), 1, f);
	int_reverse(image_num);
	assert(n == 1 && image_num == image_count);

	unsigned char* label_data = new unsigned char[image_count];

	assert(fread(label_data, sizeof(unsigned char), image_count, f) == image_count);

	fclose(f);

	return label_data;
}

unsigned char* mnist::read_image_file(const string& file, int image_count, int file_magic_num)
{
	int magic_num = 0;
	int image_num = 0;
	int row_num = 0;
	int col_num = 0;

	FILE* f = fopen(file.c_str(), "r");
	if (0 == f) {
		std::cerr << "Open \"" << file << "\" error." << std::endl;
		exit(-1);
	}

	size_t n = fread(&magic_num, sizeof(int), 1, f);
	int_reverse(magic_num);
	printf("===========magic_num:%x\t%x\n", magic_num, file_magic_num);
	assert(n == 1 && magic_num == file_magic_num);

	n = fread(&image_num, sizeof(int), 1, f);
	int_reverse(image_num);
	printf("===========image_num:%d\t%d\n", image_num, image_count);
	assert(n == 1 && image_num == image_count);

	n = fread(&row_num, sizeof(int), 1, f);
	int_reverse(row_num);
	printf("===========row_num:%d\n", row_num);
	assert(n == 1 && row_num == 28);

	n = fread(&col_num, sizeof(int), 1, f);
	int_reverse(col_num);
	printf("===========col_num:%d\n", col_num);
	assert(n == 1 && col_num == 28);

	int pixel_count = image_count * 28 * 28;;
	unsigned char* image_data = new unsigned char[pixel_count];

	assert(fread(image_data, sizeof(unsigned char), pixel_count, f) == pixel_count);

	fclose(f);

	return image_data;
}

void mnist::int_reverse(int& i) 
{
	char* p = (char*) &i;
	char ch = p[0];
	p[0] = p[3];
	p[3] = ch;
	ch = p[1];
	p[1] = p[2];
	p[2] = ch;
}

bool read_home_dir(std::string& home)
{
    const char* home_dir = NULL;
    if (NULL == (home_dir = getenv("HOME"))) {
        struct passwd* pw = getpwuid(getuid());
        if (NULL != pw)
            home_dir = pw->pw_dir;
    }

    if (home_dir) {
        home.append(home_dir);
        return true;
    } else {
        return false;
    }
}

int main(int argc, char** argv)
{
    std::string home;
    if (!read_home_dir(home)) {
        std::cerr << "Get home dir error." << std::endl;
        exit(-1);
    }
    home.append("/data/web-data/mnist-unzip/");

	mnist m(home.c_str());
	m.load_data();
	m.set_batch_size(1000);
	m.construct_net();
}
