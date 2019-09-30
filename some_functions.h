//
// Created by root on 04.09.19.
//

#include <armadillo>
#include <math.h>
#include <cstdlib>
#include <ctime>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <time.h>
#include <torch/torch.h>

using namespace arma;
using namespace std;

inline mat random_hamiltonian(double U, int L, int t) {
    int size_matrix = L*L;
    vec lista(size_matrix);
    mat lattice(size_matrix, size_matrix);
    lattice.zeros();
    // ordered list of terms with U and without U
    for(int i=0; i<size_matrix/2; i++){
        lista(i) = U;
    }
    for(int i=size_matrix/2; i<size_matrix; i++){
        lista(i) = 0;
    }
    vec matrix_shuffled = shuffle(lista); // shuffled elements

    for(int i=0; i<size_matrix; i++) lattice(i, i) = matrix_shuffled(i);
    // hopping integral

    for(int i = 0; i<size_matrix-L; i++) {
        lattice(i, i+L) = -t;
        lattice(i+L, i) = -t;
    }

    for(int i=0; i<L; i++) {
        lattice(i, i+size_matrix-L) = -t;
        lattice(i+size_matrix-L, i) = -t;
    }

    for(int i=0; i<=L-1; i++){
        lattice(i*L, i*L+L-1) = -t;
        lattice(i*L+L-1,  i*L) = -t;
    }

    for(int i=1; i<=L-1; i++) {
        for(int j=0; j<=L-1; j++) {
            lattice(i+j*L-1, i+j*L) = -t;
            lattice(i+j*L, i+j*L-1) = -t;
        }
    }
    return lattice;
}


inline void save_conf(mat hamiltonian, ofstream & file) {
    int size = hamiltonian.n_rows;
    for(int i = 0; i<size; i++) {
        if(hamiltonian(i, i) >0.0001) {
            file << 1 << " ";
        }
        else {
            file << 0 << " ";
        }
    }
    file << "\n";
}


inline mat Swap_sites(mat lattice) {
    double r;
    int lattice_n1, lattice_n2;
    int size = lattice.n_rows;
    double swap_var;

    do {
        lattice_n1 = (rand() % static_cast<int>(size -1 + 1));
    } while(lattice(lattice_n1, lattice_n1) < 0.00001);

    do {
        lattice_n2 = (rand() % static_cast<int>(size -1 + 1));

    } while(lattice(lattice_n2, lattice_n2) > 0.1);


    swap_var = lattice(lattice_n1, lattice_n1);
    lattice(lattice_n1, lattice_n1) = lattice(lattice_n2, lattice_n2);
    lattice(lattice_n2, lattice_n2) = swap_var;

    return lattice;

}



inline double energy_conf(mat hamiltonian, double beta, double cp) {
    mat eigvec;
    vec eigval;
    eig_sym(eigval, eigvec, hamiltonian);
    int N = hamiltonian.n_rows;
    double E = 0;
    double T = 1.0/beta;
    for(int j=0; j<N; j++){
        E += log(1+exp(-beta*(eigval(j)-cp)));
    }
    return -T*E;
}


inline mat saveConf(mat conf, mat ham, double U, int k) {
    int n = ham.n_rows;
    for(int i=0; i<n; i++) {
        conf(k, i) = ham(i, i)/U;
    }
    return conf;
}


inline mat MC(mat initial_hamiltonian, int MC_steps, double beta, double cp) {
    int acc = 0;
    double E0, E_new, delta, r;
    mat new_hamiltonian;
    E0 = energy_conf(initial_hamiltonian, beta, cp);
    for (int i = 0; i < MC_steps; i++) {
        new_hamiltonian = Swap_sites(initial_hamiltonian);
        E_new = energy_conf(new_hamiltonian, beta, cp);
        delta = E_new - E0;
        if (delta < 0) {
            initial_hamiltonian = new_hamiltonian;
            E0 = E_new;
            acc++;
        }

        else {
            r = ((double) rand() / (RAND_MAX));
            if (exp(-delta * beta) >= r) {
                initial_hamiltonian = new_hamiltonian;
                E0 = E_new;
                acc++;
            }
        }
    }
    //cout << "acc= " << acc << endl;
    return initial_hamiltonian;

}

// SIMPLE NEURAL NETWORKS


struct ModelNN : torch::nn::Module {

    ModelNN() {
        in = register_module("in",torch::nn::Linear(100,100));
        h = register_module("h",torch::nn::Linear(100,50));
        out = register_module("out",torch::nn::Linear(50,1));
    }

    torch::Tensor forward(torch::Tensor X){
        X = torch::relu(in->forward(X));
        X = torch::relu(h->forward(X));
        X = out->forward(X);
        return X;
    }

    torch::nn::Linear in{nullptr}, h{nullptr}, out{nullptr};

};


struct ModelCNN : torch::nn::Module {
    ModelCNN()
            : conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
              conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
              fc1(320, 50),
              fc2(50, 10) {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv2_drop", conv2_drop);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(
                torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = x.view({-1, 320});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }

    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::FeatureDropout conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};


inline torch::Tensor VecToTensor(vec data) {
    int n = data.n_rows;
    torch::Tensor tensor = torch::ones({n});

    for(int i=0; i<n; i++) {
        tensor[i] = data(i);
    }
    cout << tensor.sizes() << endl;
    return tensor;
}

inline torch::Tensor MatToTensor(mat data) {
    int n = data.n_rows;
    int m = data.n_cols;

    torch::Tensor tensor = torch::ones({n, m});

    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            tensor[i][j] = data(i, j);
        }
    }
    return tensor;
}


torch::Tensor mat_read_data(const std::string& loc)
{
    mat conf;
    conf.load(loc);
    torch::Tensor tensor = MatToTensor(conf);

    return tensor;
};

torch::Tensor vec_read_data(const std::string& loc) {
    vec energies;
    energies.load(loc);
    torch::Tensor tensor = VecToTensor(energies);

    return tensor;
}




class MyDataset : public torch::data::Dataset<MyDataset>
{
private:
    torch::Tensor states_, labels_;

public:
    explicit MyDataset(const std::string& loc_states, const std::string& loc_labels)
            : states_(mat_read_data(loc_states)),
              labels_(vec_read_data(loc_labels)) {   };

    torch::data::Example<> get(size_t index) override {
        return {states_[index], labels_[index]};

    }
    torch::optional<size_t> size() const override {
        return states_.sizes()[0];
    }

};

// SLMC

mat confToHam (rowvec conf, double U, int t, int L) {
    int size_matrix = conf.n_cols;
    cout << size_matrix << endl;
    mat lattice(size_matrix, size_matrix);
    lattice.zeros();

    for(int i=0; i<size_matrix; i++) lattice(i, i) = conf(i)*U;

    for(int i = 0; i<size_matrix-L; i++) {
        lattice(i, i+L) = -t;
        lattice(i+L, i) = -t;
    }

    for(int i=0; i<L; i++) {
        lattice(i, i+size_matrix-L) = -t;
        lattice(i+size_matrix-L, i) = -t;
    }

    for(int i=0; i<=L-1; i++){
        lattice(i*L, i*L+L-1) = -t;
        lattice(i*L+L-1,  i*L) = -t;
    }

    for(int i=1; i<=L-1; i++) {
        for(int j=0; j<=L-1; j++) {
            lattice(i+j*L-1, i+j*L) = -t;
            lattice(i+j*L, i+j*L-1) = -t;
        }
    }
    return lattice;
}

mat LastHam (mat data, double U, int t, int L) {

    int D = data.n_rows;
    rowvec LastConf = data.row(D-1);
    mat LastHam = confToHam(LastConf, U, t, L);

    return LastHam;
}

double energySLMC (mat hamiltonian, ModelNN model) {

}

mat SLMC_effective (mat initial_hamiltonian, double beta, int MC_steps, ModelNN model) {
    double E0 = energySLMC(initial_hamiltonian, model);
    double E_new, delta, r;
    mat new_hamiltonian;

    for(int i=0; i<MC_steps; i++) {
        new_hamiltonian = Swap_sites(initial_hamiltonian);
        E_new = energySLMC(new_hamiltonian, model);
        delta = E_new - E0;
        if(delta < 0) {
            initial_hamiltonian = new_hamiltonian;
            E0 = E_new;
        }
        else {
            r = ((double) rand() / (RAND_MAX));
            if(exp(-beta*delta) >= r) {
                initial_hamiltonian = new_hamiltonian;
                E0 = E_new;
            }
        }
    }

    return initial_hamiltonian;
}

mat SLMC(mat initial_hamiltonian, double beta, double cp, int n_conf, int MC_steps, ModelNN model) {
    double EA = energy_conf(initial_hamiltonian, beta, cp);
    double EA_eff = energySLMC(initial_hamiltonian, model);
    double EB, EB_eff, delta, r;
    mat new_hamiltonian;

    for(int i=0; i<n_conf; i++) {
        new_hamiltonian = SLMC_effective(initial_hamiltonian, beta, MC_steps, model);
        EB = energy_conf(new_hamiltonian, beta, cp);
        EB_eff = energySLMC(new_hamiltonian, model);

        delta = EB - EA  + EA_eff - EB_eff;
        if(delta < 0) {
            initial_hamiltonian = new_hamiltonian;
            EA = EB;
            EA_eff = EB_eff;
        }
        else {
            r = ((double) rand() / (RAND_MAX));
            if(exp(-beta*delta) >= r) {
                initial_hamiltonian = new_hamiltonian;
                EA = EB;
                EA_eff = EB_eff;
            }
        }
    }

    return initial_hamiltonian;
}






