
#include <pybind11/pybind11.h>
#include "knn.h"
#include "eig.h"
#include "furthest_point_sampling.h"
#include "ball_query.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eig", &eig);
    m.def("knn", &knn);
    m.def("furthest_point_sampling", &furthest_point_sampling);
    m.def("ball_query", &ball_query);
}
