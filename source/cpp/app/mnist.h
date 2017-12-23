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

#include "matrix0.h"
#include "net.h"
#include <string>

using std::string;
class mnist {
	public:
		const char* path_prefix;
	private:
		string train_image_file;
		string train_label_file;
		matrix* train_image;
		matrix* train_label;
		matrix* valid_image;
		matrix* valid_label;
		string test_image_file;
		string test_label_file;
		matrix* test_image;
		matrix* test_label;
	public:
		mnist(const char* path_prefix);
		void load_data();


		inline void set_batch_size(int bz) {
			batch_size = bz;
		}

		void construct_net();
	private:
		unsigned char* read_label_file(const string& file, int image_count, int file_magic_num);
		unsigned char* read_image_file(const string& file, int image_count, int file_magic_num);
		void int_reverse(int& i);
		void shrink_image(unsigned char* orig, float* vec);
	private:
		int batch_size;
		net* mnist_net;
};
