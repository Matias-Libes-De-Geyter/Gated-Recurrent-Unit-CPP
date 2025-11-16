#include "TrainerClassifier.hpp"

TrainerClassifier::TrainerClassifier(GRU& model, const hyperparameters& hyper) : _model(model), _hyper(hyper) {
	_scope = nullptr;
	_xtrain = nullptr;
	_ytrain = nullptr;
	_xvalid = nullptr;
	_yvalid = nullptr;
}

void TrainerClassifier::set_scope(Scope& scope) {
	_scope = &scope;
}

void TrainerClassifier::set_data(Dataset& train, Dataset& validation) {
	_xtrain = &train.x;
	_ytrain = &train.y;

	_xvalid = &validation.x;
	_yvalid = &validation.y;
}

void TrainerClassifier::run(const bool store) {

	// Store data init
	d_vector CELoss;
	d_vector train_acc_array;
	d_vector val_acc_array;

	// Early stopping
	int nb_epochs = _hyper.max_epochs;

	const int n_batches = _hyper.n_batch / _hyper.batch_size;
	for (int epoch = 0; epoch < nb_epochs; epoch++) {

		double epoch_loss = 0;
		int train_correct = 0;
		int val_correct = 0;

		// Train accuracy
		for (int n = 0; n < n_batches; n++) {
			const std::vector<Matrix>& X = (*_xtrain)[n];
			const std::vector<Matrix>& Y = (*_ytrain)[n];

			_model.forward(X);
			_model.backpropagation(X, Y);

			_scope->step(_model);

			// Loss & Accuracy computation
			const std::vector<Matrix>& y_pred = _model.getOutput();
			epoch_loss += sequence_loss(_model.getOutput(), Y);
			for (int i = 0; i < _hyper.seq_len; i++)
				for(int j = 0; j < _hyper.batch_size; j++)
					if (y_pred[i].getBinary()(j, 0) == Y[i](j, 0))
						train_correct++;
		}

		// Validation accuracy
		for (int n = 0; n < _hyper.test_size; n++) {
			const std::vector<Matrix>& X = (*_xvalid)[n];
			const std::vector<Matrix>& Y = (*_yvalid)[n];

			_model.forward(X);

			// Accuracy computation
			const std::vector<Matrix>& y_pred = _model.getOutput();
			for (int i = 0; i < _hyper.seq_len; i++)
				if (y_pred[i].getBinary() == Y[i])
					val_correct++;
		}


		epoch_loss /= _hyper.n_batch;
		double train_accuracy = 100.0 * train_correct / _hyper.n_batch / _hyper.seq_len;
		double val_accuracy = 100.0 * val_correct / _hyper.test_size / _hyper.seq_len;

		// Printing the results
		print("[Epoch ", epoch + 1, "/", _hyper.max_epochs, "] ",
			"Loss = ", epoch_loss, " | ",
			"train_acc = ", train_accuracy, " % | ",
			"test_acc = ", val_accuracy, " %");


		// Storing data
		if (store) {
			// Accuracy
			train_acc_array.push_back(train_accuracy);
			val_acc_array.push_back(val_accuracy);

			// Loss
			CELoss.push_back(epoch_loss);
		}

		// Early stopper (very primitive for now. Might have to implement it in the Scope for better readability)
		if (val_accuracy > 99 && train_accuracy > 99) {
			// break only after four perfect iterations.
			static int i = 0; if (i == 3) nb_epochs = epoch; i++;
		}
	}
	if (store)
		writeFile(train_acc_array, val_acc_array, CELoss, nb_epochs, "training_data.csv");
}