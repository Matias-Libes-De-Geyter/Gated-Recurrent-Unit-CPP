#include "..\Utilities/functions.hpp"


#pragma once

#ifndef DATASET_HPP
#define DATASET_HPP

// Dataset structure
struct Dataset {
	std::vector<std::vector<Matrix>> x;
	std::vector<std::vector<Matrix>> y;
};

// Loading the data into the struct
inline Dataset DataLoader(const hyperparameters& hyper, const std::string& dataset_type) {

	int n_iter = 0;
	int batch_size = 0;
	// For training, we have batch_size batch
	if (dataset_type == "train") {
		n_iter = hyper.n_batch / hyper.batch_size;
		batch_size = hyper.batch_size;
	}
	// For testing, we have only 1 batch.
	else if (dataset_type == "test") {
		n_iter = hyper.test_size;
		batch_size = 1;
	}
	else
		print("Dataset type is wrong");


	Dataset data;
	data.x.reserve(n_iter);
	data.y.reserve(n_iter);

	for (int n = 0; n < n_iter; n++) {

		std::vector<Matrix> x_batch;
		x_batch.reserve(hyper.seq_len);
		std::vector<Matrix> y_batch;
		y_batch.reserve(hyper.seq_len);

		std::vector<int> sum(batch_size, 0);

		// Fill in X with the random bits, and Y with the parity of the bits until then
		for (int i = 0; i < hyper.seq_len; i++) {

			Matrix Xt(batch_size, 1);
			for (int b = 0; b < batch_size; b++) {
				Xt(b, 0) = random_bit();
				sum[b] += Xt(b, 0);
			}
			x_batch.emplace_back(std::move(Xt));

			Matrix Yt(batch_size, 1);
			for (int b = 0; b < batch_size; ++b)
				Yt(b, 0) = sum[b] % 2;
			y_batch.emplace_back(std::move(Yt));
		}

		// Emplace the data into the vectors
		data.x.emplace_back(std::move(x_batch));
		data.y.emplace_back(std::move(y_batch));
	}

	return data;
};

#endif