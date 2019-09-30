#include <armadillo>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "some_functions.h" // helping functions
#include <time.h>
#include <torch/torch.h>

using namespace std;
using namespace arma;


template <typename DataLoader>

void train(size_t epoch, ModelNN& model,
        torch::Device device,
        DataLoader& data_loader,
        torch::optim::Optimizer& optimizer,
        size_t dataset_size) {

}

int main() {

    // 1.  MONTE CARLO

    // a) variables

    int L = 10;
    int t = 1;
    int n = 1; // the multiple of space between the independent configurations
    int space = L * L * n; // space between the independent configurations
    int mes = 10000;
    int therm_steps = 4000;
    double U = 4.0;
    double cp = U / 2; //chemical potential
    double T = 0.2;
    double beta = 1.0 / T;
    double E_new;

//    mat new_hamiltonian = random_hamiltonian(U, L, t);
//    mat conf_train(mes, L*L);
//    vec energy_train(mes);
//    double E0 = energy_conf(new_hamiltonian, beta, cp);

    struct timespec tstart={0,0}, tend={0,0};
    double diff;

//    ofstream conf;
//    ofstream histogram_energies;
//
//
////    // b) thermalisation
//
//    cout << "Poczatek termalizacji" << endl;
//
//    new_hamiltonian = MC(new_hamiltonian, therm_steps, beta, cp);
//
//    cout<<"koniec termalizacji"<<endl;

    const string X_train = "conf_train.txt";
    const string y_train = "energy_train.txt";

    const string X_test = "conf_test.txt";
    const string y_test = "energy_test.txt";


    // c) Metropolis

//    for (int k=0; k<mes; k++) {
//        if(k%100 == 0) {
//            cout << k << " konfiguracji" << endl;
//        }
//        new_hamiltonian = MC(new_hamiltonian, space, beta, cp);
//        E_new = energy_conf(new_hamiltonian, beta, cp);
//        conf_train = saveConf(conf_train, new_hamiltonian, U, k);
//        energy_train(k) = E_new;
//    }
//
//    conf_train.save(X_test, arma_ascii);
//    energy_train.save(y_test, arma_ascii);


// MACHINE LEARNING
// A) DATA LOAD

    const int64_t kTrainBatchSize = 128;
    const int64_t kTestBatchSize = 1000;
    const int64_t epochs = 30;



    auto data_set = MyDataset(X_train, y_train).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(data_set), kTrainBatchSize);


    cout << "ok" << endl;

    // B) ACTUAL TRAINING

    torch::manual_seed(1);

    torch::DeviceType device_type;
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    } else {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);


    ModelNN model;
    model.to(device);
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    model.train();

    size_t batch_idx = 0;
    float value_loss;
    for(int i=0; i<epochs; i++) {
        cout << i << endl;
        for (auto &batch : *data_loader) {
            auto data = batch.data.to(device), targets = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model.forward(data);
            auto loss = torch::mse_loss(output.squeeze(), targets);
            value_loss = loss.template item<float>();
            AT_ASSERT(!std::isnan(loss.template item<float>()));
            loss.backward();
            optimizer.step();
            }

        printf("\rTrain Epoch: %d Loss: %.4f",
                i, value_loss);
    }



// SIMULATION OF THE EFFECTIVE MODEL


//    mat Training;
//    Training.load(X_train);


    return 0;
}