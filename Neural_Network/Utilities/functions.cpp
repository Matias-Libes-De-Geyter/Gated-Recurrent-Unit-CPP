#include "functions.hpp"

// Random function
std::mt19937_64& get_rng() {
	static std::mt19937_64 rng{ std::random_device{}() };
	return rng;
}
double random(const double& min, const double& max) {
	return std::uniform_real_distribution<>{min, max}(get_rng());
}
int random_bit() {
	static std::uniform_int_distribution<int> dist(0, 1);
	return dist(get_rng());
}

// Cross-entropy loss computation
double sequence_loss(const std::vector<Matrix>& y_pred_seq, const std::vector<Matrix>& y_true_seq) {
    assert(y_pred_seq.size() == y_true_seq.size());

    double total = 0.0;
    long total_elements = 0;

    for (int t = 0; t < y_pred_seq.size(); ++t) {

        const Matrix& p = y_pred_seq[t];
        const Matrix& y = y_true_seq[t];
		
        for (int i = 0; i < p.rows(); ++i)
            for (int j = 0; j < p.cols(); ++j)
                total += -(y(i, j) * std::log(p(i, j)) + (1.0 - y(i, j)) * std::log(1.0 - p(i, j)));
        total_elements += (long)p.rows() * p.cols();
    }

    if (total_elements == 0) return 0.0;
    return total / (double)total_elements;
}

// Write output data to plot with python
void writeFile(const d_vector& train_acc, const d_vector& val_acc, const d_vector& loss, int nb_epochs, const std::string& filename) {
	std::ofstream outFile(filename);
	if (!outFile) {
		std::cerr << "Error opening file for writing: " << filename << std::endl;
		return;
	}

	outFile << "Epoch,TrainAcc,ValAcc,Loss\n";
	for (int epoch = 0; epoch < nb_epochs; ++epoch) {
		outFile << epoch + 1 << "," << train_acc[epoch] << "," << val_acc[epoch] << "," << loss[epoch] << "\n";
	}

	outFile.close();
}