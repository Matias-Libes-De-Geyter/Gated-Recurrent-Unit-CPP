#include "GRU/GRU.hpp"
#include "Classifier/TrainerClassifier.hpp"
#include "Classifier/Scope.hpp"
#include "Dataset/Dataset.hpp"

hyperparameters hyper = {
    seq_len: 1000,
    input_dimension : 1,
    hidden_dimension : 16,
    output_dimension : 1,
    learning_rate : 0.005,
    max_epochs : 200,
    n_batch : 1000,
    batch_size : 32,
    test_size : 100
};

const bool store = true;

int main() {

    GRU model(hyper);

    Scope scope(model, hyper);

    TrainerClassifier trainer(model, hyper);

    Dataset train = DataLoader(hyper, "train");
    Dataset validation = DataLoader(hyper, "test");

    trainer.set_scope(scope);
    trainer.set_data(train, validation);
    print("Data has been successfully imported");

    trainer.run(store);

    return 0;
};
