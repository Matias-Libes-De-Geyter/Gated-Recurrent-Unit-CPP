#include "Matrix.hpp"

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> init) {
	_rows = init.size();
	_cols = init.begin()->size();

	_matrix.clear();
	_matrix.reserve(_rows * _cols);

	for (const auto& row : init)
		_matrix.insert(_matrix.end(), row.begin(), row.end());
}

Matrix& Matrix::operator=(std::initializer_list<std::initializer_list<double>> init) {
	_rows = init.size();
	_cols = init.begin()->size();

	_matrix.clear();
	_matrix.reserve(_rows * _cols);

	for (auto& row : init)
		_matrix.insert(_matrix.end(), row.begin(), row.end());

	return *this;
}

const bool Matrix::operator==(const Matrix& B) const {
	if (_rows == B.rows()) {
		if (_cols == B.cols()) {
			for (size_t idx = 0; idx < _rows * _cols; idx++)
				if (B(idx) != _matrix[idx])
					return false;
			return true;
		}
	}

	return false;
}

Matrix Matrix::operator*(const Matrix& B) const {
	assert(_cols == B.rows());

	size_t new_cols = B.cols();
	Matrix C(_rows, new_cols);

	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t k = 0; k < _cols; k++) {
			double Aik = _matrix[row_offset + k];
			for (size_t j = 0; j < new_cols; j++)
				C(i, j) += Aik * B(k, j);
		}
	}

	return C;
}


Matrix Matrix::operator*(const double b) const {

	Matrix C(_rows, _cols);

	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j] * b;
	}

	return C;
}
Matrix& Matrix::operator*=(const double b) {

	for (size_t idx = 0; idx < _rows * _cols; idx++)
		_matrix[idx] *= b;

	return *this;
}

Matrix Matrix::hadamard(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j] * B(i, j);
	}

	return C;
}


Matrix Matrix::operator+(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j] + B(i, j);
	}

	return C;
}

Matrix& Matrix::operator+=(const Matrix& B) {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			_matrix[row_offset + j] += B(i, j);
	}

	return *this;
}

Matrix Matrix::operator-(const Matrix& B) const {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j] - B(i, j);
	}

	return C;
}

Matrix& Matrix::operator-=(const Matrix& B) {
	assert(_rows == B.rows());
	assert(_cols == B.cols());

	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			_matrix[row_offset + j] -= B(i, j);
	}

	return *this;
}



Matrix Matrix::T() const {

	Matrix C(_cols, _rows);
	for (size_t j = 0; j < _rows; j++) {
		size_t row_offset = j * _cols;
		for (size_t i = 0; i < _cols; i++)
			C(i, j) = _matrix[row_offset + i];
	}

	return C;
}
Matrix Matrix::addBias() const {

	Matrix C(_rows, _cols + 1);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j];
		C(i, _cols) = 1;
	}

	return C;
}
Matrix Matrix::addBias_then_T() const {

	Matrix C(_cols + 1, _rows);
	for (size_t j = 0; j < _rows; j++) {
		size_t row_offset = j * _cols;
		for (size_t i = 0; i < _cols; i++)
			C(i, j) = _matrix[row_offset + i];
		C(_cols, j) = 1;
	}

	return C;
}
Matrix Matrix::removeBias() const {

	Matrix C(_rows - 1, _cols);
	for (size_t i = 0; i < _rows - 1; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = _matrix[row_offset + j];
	}

	return C;
}

Matrix Matrix::getBinary() const {

	Matrix C(_rows, _cols);
	for (size_t idx = 0; idx < _rows * _cols; idx++)
		C(idx) = (_matrix[idx] >= 0.5 ? 1 : 0);

	return C;
}

Matrix Matrix::oneMinus() const {

	Matrix C(_rows, _cols);
	for (size_t i = 0; i < _rows; i++) {
		size_t row_offset = i * _cols;
		for (size_t j = 0; j < _cols; j++)
			C(i, j) = 1 - _matrix[row_offset + j];
	}

	return C;
}

void Matrix::fill(const double b) {

	for (size_t idx = 0; idx < _rows * _cols; idx++)
		_matrix[idx] = b;
}