#pragma once
#include "TFEL/Math/stensor.hxx"
#include "TFEL/Math/st2tost2.hxx"
#include <tuple>

namespace neuralgsm
{

    using Stensor = tfel::math::stensor<3u, double>;
    using Stensor4 = tfel::math::st2tost2<3u, double>;

    std::tuple<Stensor, Stensor4, Stensor4>
    phi1_and_derivatives(const Stensor &, const Stensor &);

    std::tuple<Stensor, Stensor4, Stensor4>
    phi2_and_derivatives(const Stensor &, const Stensor &);

}
