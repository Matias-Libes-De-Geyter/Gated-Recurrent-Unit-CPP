#include "Matrix.hpp"

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

// Hyperparameters
struct hyperparameters {
	int seq_len;
	int input_dimension;
	int hidden_dimension;
	int output_dimension;
	double learning_rate;
	int max_epochs;
	int n_batch;
	int batch_size;
	int test_size;
};

std::mt19937_64& get_rng();
double random(const double& min, const double& max); // Random function
int random_bit(); // Random bit between 0 and 1

double sequence_loss(const std::vector<Matrix>& y_pred, const std::vector<Matrix>& y_true); // Return the cross-entropy loss.

// Utility function used in TrainerClassifier.h
void writeFile(const std::vector<double>& train_acc, const std::vector<double>& test_acc, const std::vector<double>& loss, int nb_epochs, const std::string& filename);



namespace ACTIVATION {

	inline Matrix tanh_activation(Matrix& inputs) {

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++)
				output(i, j) = std::tanh(inputs(i, j));
		return output;
	};
	inline Matrix deriv_tanh(const Matrix& a) {
		Matrix out(a.rows(), a.cols());
		for (size_t i = 0; i < a.rows(); ++i)
			for (size_t j = 0; j < a.cols(); ++j) {
				double aij = a(i, j);
				out(i, j) = 1.0 - aij * aij;
			}
		return out;
	};
	inline Matrix deriv_tanh_full(Matrix& inputs) {

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++) {
				double v = std::tanh(inputs(i, j));
				output(i, j) = 1 - v * v;
			}
		return output;
	};

	inline Matrix ReLU_activation(const Matrix& inputs) {

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++)
				output(i, j) = std::max(0.0, inputs(i, j));
		return output;
	};
	inline Matrix deriv_ReLU(const Matrix& inputs) {

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++)
				output(i, j) = (inputs(i, j) > 0.0 ? 1.0 : 0);
		return output;
	};

	inline Matrix sigmoid_activation(Matrix& inputs) {

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++)
				output(i, j) = 1.0 / (1.0 + std::exp(-inputs(i, j)));
		return output;
	};
	inline Matrix deriv_sigmoid(const Matrix& s) {
		Matrix out(s.rows(), s.cols());
		for (size_t i = 0; i < s.rows(); ++i)
			for (size_t j = 0; j < s.cols(); ++j) {
				double sij = s(i, j);
				out(i, j) = sij * (1.0 - sij);
			}
		return out;
	};
	inline Matrix deriv_sigmoid_full(const Matrix& inputs) {

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++) {
				double s = 1.0 / (1.0 + std::exp(-inputs(i, j)));
				output(i, j) = s * (1.0 - s);
			}
		return output;
	};


	inline Matrix softmax_activation(const Matrix& inputs) {

		d_vector maxs(inputs.rows());
		for (size_t i = 0; i < inputs.rows(); i++) {
			maxs[i] = inputs(i, 0);
			for (size_t j = 0; j < inputs.cols(); j++)
				if (inputs(i, j) > maxs[i])
					maxs[i] = inputs(i, j);
		}

		Matrix expvalues = inputs;
		d_vector sum_of_exps(inputs.rows(), 0);
		for (size_t i = 0; i < inputs.rows(); i++) {
			for (size_t j = 0; j < inputs.cols(); j++) {
				expvalues(i, j) = std::exp(inputs(i, j) - maxs[i]);
				sum_of_exps[i] += expvalues(i, j);
			}
		}

		Matrix output(inputs.rows(), inputs.cols());
		for (size_t i = 0; i < inputs.rows(); i++)
			for (size_t j = 0; j < inputs.cols(); j++)
				output(i, j) = expvalues(i, j) / sum_of_exps[i];
		return output;
	};
}


namespace MATRIX_OPERATION {

	// output = input.addBias * weights
	// (used in m_Z[t] = input[t].addBias() * m_inWeights + m_hiddenStates[t].addBias() * m_hiddenWeights; )
	inline Matrix addbiases_then_mult(const Matrix& input, const Matrix& weights) {

		size_t output_rows = input.rows();
		size_t output_cols = weights.cols();
		size_t middle_dim = input.cols() + 1;

		assert(middle_dim == weights.rows());

		Matrix output(output_rows, output_cols);

		for (size_t i = 0; i < output_rows; i++) {
			for (size_t k = 0; k < middle_dim; k++) {
				double input_ik = (k == input.cols() ? 1 : input(i, k));
				for (size_t j = 0; j < output_cols; j++)
					output(i, j) += input_ik * weights(k, j);
			}
		}

		return output;
	};

	// weights += input.addBiasThenT * delta
	// (used in m_dU += input[t].addBias_then_T() * m_z_deltas[t]; )
	// (used in m_dW += m_hiddenStates[t].addBias_then_T() * m_z_deltas[t]; )
	inline void compute_weigths(Matrix& weights, const Matrix& input, const Matrix& delta) {

		size_t output_rows = weights.rows();
		size_t output_cols = weights.cols();
		size_t middle_dim = input.rows();

		assert(middle_dim == delta.rows());
		assert(output_rows == input.cols() + 1);
		assert(output_cols == delta.cols());

		for (size_t i = 0; i < output_rows - 1; i++) {
			for (size_t k = 0; k < middle_dim; k++) {
				double input_ik = input(k, i);
				for (size_t j = 0; j < output_cols; j++)
					weights(i, j) += input_ik * delta(k, j);
			}
		}
		for (size_t j = 0; j < output_cols; j++)
			for (size_t k = 0; k < middle_dim; k++)
				weights(output_rows - 1, j) += delta(k, j);

	};

	// weights += input.T * delta
	// (used in m_dWout += m_hiddenStates[t + 1].T() * m_y_deltas[t]; )
	inline void compute_out_weights(Matrix& weights, const Matrix& input, const Matrix& delta) {

		size_t output_rows = weights.rows();
		size_t output_cols = weights.cols();
		size_t middle_dim = input.rows();

		assert(middle_dim == delta.rows());
		assert(output_rows == input.cols());
		assert(output_cols == delta.cols());

		for (size_t i = 0; i < output_rows; i++) {
			for (size_t k = 0; k < middle_dim; k++) {
				double input_ik = input(k, i);
				for (size_t j = 0; j < output_cols; j++)
					weights(i, j) += input_ik * delta(k, j);
			}
		}
	};

}


// ===== PRINT FUNCTIONS =====
// Multiple print
template<typename... Args>
typename std::enable_if<(sizeof...(Args) > 1), void>::type
inline print(const Args&... args) { (std::cout << ... << args) << std::endl; }

// Single print
template<typename T>
typename std::enable_if<!std::is_same<T, Matrix>::value, void>::type
inline print(const T& arg) { std::cout << arg << std::endl; }

// Matrix print
inline void print(const Matrix& A) {
	const size_t rows = A.rows();
	const size_t cols = A.cols();

	std::ostringstream stream;
	stream << "[";
	for (size_t i = 0; i < rows; i++) {
		stream << "[";
		for (size_t j = 0; j < cols; j++) {
			stream << A(i, j);
			if (j < cols - 1) stream << ", ";
		}
		stream << "]";
		if (i < rows - 1) stream << ", " << std::endl;
	}
	stream << "]" << std::endl;

	std::cout << stream.str();
}

#endif