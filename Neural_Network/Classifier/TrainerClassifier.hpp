#include "Scope.hpp"
#include "..\Dataset/Dataset.hpp"


#ifndef TRAINER_HPP
#define TRAINER_HPP

class TrainerClassifier {
private:
	const hyperparameters _hyper;
	GRU& _model;

	Scope* _scope;
	std::vector<std::vector<Matrix>>* _xtrain;
	std::vector<std::vector<Matrix>>* _ytrain;
	std::vector<std::vector<Matrix>>* _xvalid;
	std::vector<std::vector<Matrix>>* _yvalid;

public:
	// Constructor, set methods and run method.
	TrainerClassifier(GRU&, const hyperparameters&);
	void set_scope(Scope&);
	void set_data(Dataset&, Dataset&);
	void run(const bool store);
};

#endif