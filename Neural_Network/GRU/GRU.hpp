#include "..\Utilities/functions.hpp"

#ifndef GRU_HPP
#define GRU_HPP

class GRU {

private:
	const hyperparameters& _hyper;

	// Weights, values matrices and their gradient.
	Matrix m_Ur;
	Matrix m_Uhat;
	Matrix m_Uz;
	Matrix m_Wz;
	Matrix m_Wr;
	Matrix m_What;
	Matrix m_Wout;

	Matrix m_dUr;
	Matrix m_dUhat;
	Matrix m_dUz;
	Matrix m_dWz;
	Matrix m_dWr;
	Matrix m_dWhat;
	Matrix m_dWout;

	std::vector<Matrix> m_aZ;
	std::vector<Matrix> m_aR;
	std::vector<Matrix> m_aHat;

	std::vector<Matrix> m_hiddenStates;
	std::vector<Matrix> m_Y;

	std::vector<Matrix> m_dZ;
	std::vector<Matrix> m_dR;
	std::vector<Matrix> m_dHat;
	std::vector<Matrix> m_dH;
	std::vector<Matrix> m_dY;


public:
	GRU(const hyperparameters& hyper);

	// Forward and Backprop
	void forward(const std::vector<Matrix>& input);
	void backpropagation(const std::vector<Matrix>& input, const std::vector<Matrix>& y_real);

	// Return the output vector
	inline const std::vector<Matrix>& getOutput() const {
		return m_Y;
	};

	// Return all the diff parameters and their gradient for later-on optimization (in Scope)
	inline std::vector<std::pair<Matrix*, Matrix*>> getParameters() {
		return {
		{ &m_Uz, &m_dUz },
		{ &m_Ur, &m_dUr },
		{ &m_Uhat, &m_dUhat },
		{ &m_Wz, &m_dWz },
		{ &m_Wr, &m_dWr },
		{ &m_What, &m_dWhat },
		{ &m_Wout, &m_dWout } };
	};

};

#endif