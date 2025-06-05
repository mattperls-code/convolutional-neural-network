#ifndef TESTS_UTIL_HPP
#define TESTS_UTIL_HPP

#include "../src/tensor.hpp"

void prettyPrintMatrix(const Matrix& mat);

void prettyPrintTensor(const Tensor& tensor);

bool matricesAreApproxEqual(const Matrix& matA, const Matrix& matB, float margin);

bool tensorsAreApproxEqual(const Tensor& tensorA, const Tensor& tensorB, float margin);

#endif