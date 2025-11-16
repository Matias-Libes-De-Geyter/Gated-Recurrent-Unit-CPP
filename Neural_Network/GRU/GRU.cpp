#include "GRU.hpp"


GRU::GRU(const hyperparameters& hyper) : _hyper(hyper) {

	m_Wz = Matrix(hyper.input_dimension + 1, hyper.hidden_dimension);
	m_Wr = m_Wz;
	m_What = m_Wz;
	m_Uz = Matrix(hyper.hidden_dimension + 1, hyper.hidden_dimension);
	m_Ur = m_Uz;
	m_Uhat = m_Uz;
	m_Wout = Matrix(_hyper.hidden_dimension, _hyper.output_dimension);

	m_dUz = m_Uz;
	m_dUr = m_Uz;
	m_dUhat = m_Uz;
	m_dWz = m_Wz;
	m_dWr = m_Wz;
	m_dWhat = m_Wz;
	m_dWout = m_Wout;

	double limit = 0.6;

	// inW and outW are initialised as already transposed, for better performance
	for (size_t i = 0; i < _hyper.hidden_dimension; i++)
		for (size_t j = 0; j < _hyper.input_dimension + 1; j++) {
			m_Wz(j, i) = random(-limit, limit);
			m_Wr(j, i) = random(-limit, limit);
			m_What(j, i) = random(-limit, limit);
		}
	for (size_t i = 0; i < _hyper.hidden_dimension; i++)
		for (size_t j = 0; j < _hyper.hidden_dimension + 1; j++) {
			m_Uz(j, i) = random(-limit, limit);
			m_Ur(j, i) = random(-limit, limit);
			m_Uhat(j, i) = random(-limit, limit);
		}
	for (size_t i = 0; i < _hyper.output_dimension; i++)
		for (size_t j = 0; j < _hyper.hidden_dimension; j++)
			m_Wout(j, i) = random(-limit, limit);

	m_aZ.clear();
	m_aZ.resize(_hyper.seq_len);
	m_aR.clear();
	m_aR.resize(_hyper.seq_len);
	m_aHat.clear();
	m_aHat.resize(_hyper.seq_len);
	m_hiddenStates.clear();
	m_hiddenStates.resize(_hyper.seq_len + 1);
	m_Y.clear();
	m_Y.resize(_hyper.seq_len);

	m_dY.clear();
	m_dY.resize(_hyper.seq_len);
	m_dH.clear();
	m_dH.resize(_hyper.seq_len);
	m_dZ.clear();
	m_dZ.resize(_hyper.seq_len);
	m_dR.clear();
	m_dR.resize(_hyper.seq_len);
	m_dHat.clear();
	m_dHat.resize(_hyper.seq_len);

}

// Forward method
void GRU::forward(const std::vector<Matrix>& input) {

	m_hiddenStates[0] = Matrix(input[0].rows(), _hyper.hidden_dimension);

	// For each time step, Ht+1 = GRU FORMULA && Yt = a(Ht+1 * Wout)
	for (int t = 0; t < _hyper.seq_len; t++) {
		
		Matrix Z =
			MATRIX_OPERATION::addbiases_then_mult(input[t], m_Wz) +
			MATRIX_OPERATION::addbiases_then_mult(m_hiddenStates[t], m_Uz);
		m_aZ[t] = ACTIVATION::sigmoid_activation(Z);
		Matrix R =
			MATRIX_OPERATION::addbiases_then_mult(input[t], m_Wr) +
			MATRIX_OPERATION::addbiases_then_mult(m_hiddenStates[t], m_Ur);
		m_aR[t] = ACTIVATION::sigmoid_activation(R);
		Matrix Hat =
			MATRIX_OPERATION::addbiases_then_mult(input[t], m_What) +
			MATRIX_OPERATION::addbiases_then_mult(m_aR[t].hadamard(m_hiddenStates[t]), m_Uhat);
		m_aHat[t] = ACTIVATION::tanh_activation(Hat);

		Matrix H = (m_aZ[t].oneMinus()).hadamard(m_hiddenStates[t]) + m_aZ[t].hadamard(m_aHat[t]);
		m_hiddenStates[t + 1] = H;

		Matrix y = m_hiddenStates[t + 1] * m_Wout;
		m_Y[t] = ACTIVATION::sigmoid_activation(y);
	}
}


// Backpropagation method. No optimization, just updating the gradients.
void GRU::backpropagation(const std::vector<Matrix>& input, const std::vector<Matrix>& y_real) {

	m_dUz.fill(0.0);
	m_dUr.fill(0.0);
	m_dUhat.fill(0.0);
	m_dWz.fill(0.0);
	m_dWr.fill(0.0);
	m_dWhat.fill(0.0);
	m_dWout.fill(0.0);

	for (int t = _hyper.seq_len - 1; t >= 0; t--) {

		m_dY[t] = (m_Y[t] - y_real[t]);
		Matrix dH = m_dY[t] * m_Wout.T();

		if (t < _hyper.seq_len - 1)
			dH += m_dH[t + 1];

		Matrix d_aHat = dH.hadamard(m_aZ[t]);
		m_dHat[t] = d_aHat.hadamard(ACTIVATION::deriv_tanh(m_aHat[t])); 
		MATRIX_OPERATION::compute_weigths(m_dWhat, input[t], m_dHat[t]);
		MATRIX_OPERATION::compute_weigths(m_dUhat, m_aR[t].hadamard(m_hiddenStates[t]), m_dHat[t]);

		Matrix d_from_hat = m_dHat[t] * m_Uhat.removeBias().T();
		Matrix d_hidden_via_hat = d_from_hat.hadamard(m_aR[t]);

		Matrix dz_part = dH.hadamard(m_aHat[t] - m_hiddenStates[t]);
		m_dZ[t] = dz_part.hadamard(ACTIVATION::deriv_sigmoid(m_aZ[t]));
		MATRIX_OPERATION::compute_weigths(m_dWz, input[t], m_dZ[t]);
		MATRIX_OPERATION::compute_weigths(m_dUz, m_hiddenStates[t], m_dZ[t]);

		Matrix dr_from_hat = d_from_hat.hadamard(m_hiddenStates[t]);
		m_dR[t] = dr_from_hat.hadamard(ACTIVATION::deriv_sigmoid(m_aR[t]));
		MATRIX_OPERATION::compute_weigths(m_dWr, input[t], m_dR[t]);
		MATRIX_OPERATION::compute_weigths(m_dUr, m_hiddenStates[t], m_dR[t]);

		MATRIX_OPERATION::compute_out_weights(m_dWout, m_hiddenStates[t + 1], m_dY[t]);

		Matrix contrib_via_one_minus_z = dH.hadamard(m_aZ[t].oneMinus());
		Matrix contrib_via_Uz = m_dZ[t] * m_Uz.removeBias().T();
		Matrix contrib_via_Ur = m_dR[t] * m_Ur.removeBias().T();

		m_dH[t] = d_hidden_via_hat + contrib_via_one_minus_z + contrib_via_Uz + contrib_via_Ur;
	}

	// Normalizing the gradients
	double norm = 1.0 / (_hyper.batch_size * _hyper.seq_len);
	m_dUz *= norm;
	m_dUr *= norm;
	m_dUhat *= norm;
	m_dWz *= norm;
	m_dWr *= norm;
	m_dWhat *= norm;
	m_dWout *= norm;
}