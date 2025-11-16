#include "..\GRU/GRU.hpp"

#ifndef SCOPE_HPP
#define SCOPE_HPP

class Scope {
private:
	const hyperparameters& _hyper;

	// M, V matrices for Adam optimizer
	std::vector<Matrix> M, V;

	// Time index for Adam optimizer
	int t;

public:
	// Constructor method
	Scope(GRU&, const hyperparameters&);

	// Optimizers
	void Adam(Matrix& W, Matrix& dW, const int k);
	void SGD(Matrix& W, Matrix& dW);

	// 'Update parameters' method using the chosen optimizer.
	inline void step(GRU& model) {
		int k = 0;
		for (auto& [W, dW] : model.getParameters())
			Adam(*W, *dW, k++);

		t++;
	};

};

#endif