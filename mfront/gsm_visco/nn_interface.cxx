#include "nn_interface.hxx"
#include "cppflow/cppflow.h"

namespace neuralgsm
{

    struct NN
    {
        explicit NN(const std::string &path) : model(path) {}
        cppflow::model model;
    };

    static NN &getNN()
    {
        static NN nn("../../model/new_model");
        return nn;
    }

    std::tuple<Stensor, Stensor4, Stensor4>
    phi1_and_derivatives(const Stensor &eps, const Stensor &alpha)
    {
        auto &nn = getNN();

        cppflow::tensor epsilon_tf(
            std::vector<double>(eps.begin(), eps.end()), {6});
        cppflow::tensor alpha_tf(
            std::vector<double>(alpha.begin(), alpha.end()), {6});

        auto outputs = nn.model(
            {{"phi1_epsilon:0", epsilon_tf},
             {"phi1_alpha:0", alpha_tf}},
            {"PartitionedCall:2", "PartitionedCall:1", "PartitionedCall:0"});

        const cppflow::tensor &phi1_tf = outputs[0];
        const cppflow::tensor &dphi1_dx_tf = outputs[1];
        const cppflow::tensor &dphi1_dy_tf = outputs[2];

        Stensor phi1_val;
        auto data_phi1 = phi1_tf.get_data<double>();
        for (size_t i = 0; i < 6; ++i)
            phi1_val[i] = data_phi1[i];

        Stensor4 dphi1_dx;
        auto data_dx = dphi1_dx_tf.get_data<double>();
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < 6; ++j)
                dphi1_dx(i, j) = data_dx[i * 6 + j];

        Stensor4 dphi1_dy;
        auto data_dy = dphi1_dy_tf.get_data<double>();
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < 6; ++j)
                dphi1_dy(i, j) = data_dy[i * 6 + j];

        return {phi1_val, dphi1_dx, dphi1_dy};
    }

    std::tuple<Stensor, Stensor4, Stensor4>
    phi2_and_derivatives(const Stensor &eps, const Stensor &alpha)
    {
        auto &nn = getNN();

        cppflow::tensor epsilon_tf(
            std::vector<double>(eps.begin(), eps.end()), {6});
        cppflow::tensor alpha_tf(
            std::vector<double>(alpha.begin(), alpha.end()), {6});

        auto outputs = nn.model(
            {{"phi2_epsilon:0", epsilon_tf},
             {"phi2_alpha:0", alpha_tf}},
            {"PartitionedCall_1:0", "PartitionedCall_1:2", "PartitionedCall_1:1"});

        const cppflow::tensor &phi2_tf = outputs[0];
        const cppflow::tensor &dphi2_dx_tf = outputs[1];
        const cppflow::tensor &dphi2_dy_tf = outputs[2];

        Stensor phi2_val;
        auto data_phi2 = phi2_tf.get_data<double>();
        for (size_t i = 0; i < 6; ++i)
            phi2_val[i] = data_phi2[i];

        Stensor4 dphi2_dx;
        auto data_dx = dphi2_dx_tf.get_data<double>();
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < 6; ++j)
                dphi2_dx(i, j) = data_dx[i * 6 + j];

        Stensor4 dphi2_dy;
        auto data_dy = dphi2_dy_tf.get_data<double>();
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < 6; ++j)
                dphi2_dy(i, j) = data_dy[i * 6 + j];

        return {phi2_val, dphi2_dx, dphi2_dy};
    }

} // namespace neuralgsm
