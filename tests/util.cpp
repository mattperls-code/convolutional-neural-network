#include <catch2/catch_all.hpp>
#include <iostream>

#include "util.hpp"

void prettyPrintMatrix(const Matrix& matrix)
{
    for (int r = 0;r<matrix.rowCount();r++) {
        for (int c = 0;c<matrix.colCount();c++) {
            std::cout << matrix.get(r, c) << "\t";
        }

        std::cout << std::endl << std::endl;
    }
};

void prettyPrintTensor(const Tensor& tensor)
{
    for (int i = 0;i<tensor.getDimensions().depth;i++) {
        std::cout << "------------------" << std::endl << std::endl;

        prettyPrintMatrix(tensor.getMatrix(i));
    }
};

bool matricesAreApproxEqual(const Matrix& matA, const Matrix& matB, float margin)
{
    if (matA.shape() != matB.shape()) {
        std::cout << "matA:" << std::endl;
        prettyPrintMatrix(matA);

        std::cout << std::endl;

        std::cout << "matB:" << std::endl;
        prettyPrintMatrix(matB);

        return false;
    };
    
    for (int r = 0;r<matA.rowCount();r++) for (int c = 0;c<matA.colCount();c++) if (matA.get(r, c) != Catch::Approx(matB.get(r, c)).margin(margin)) {  
        std::cout << "matA:" << std::endl;
        prettyPrintMatrix(matA);

        std::cout << std::endl;

        std::cout << "matB:" << std::endl;
        prettyPrintMatrix(matB);

        return false;
    };

    return true;
};

bool tensorsAreApproxEqual(const Tensor& tensorA, const Tensor& tensorB, float margin)
{
    if (tensorA.getDimensions() != tensorB.getDimensions()) {
        std::cout << "tensorA:" << std::endl;
        prettyPrintTensor(tensorA);

        std::cout << std::endl;

        std::cout << "tensorB:" << std::endl;
        prettyPrintTensor(tensorB);

        return false;
    }

    for (int i = 0;i<tensorA.getDimensions().depth;i++) {
        if (!matricesAreApproxEqual(tensorA.getMatrix(i), tensorB.getMatrix(i), margin)) {
            std::cout << "at depth " << i << std::endl;

            return false;
        }
    }

    return true;
};