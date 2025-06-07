#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <string>
#include <random>

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

class Shape
{
    public:
        int rows = 0;
        int cols = 0;

        Shape() = default;
        Shape(int rows, int cols): rows(rows), cols(cols) {};

        bool operator==(const Shape&) const = default;

        int area() const;

        std::string toString() const;

        template <class Archive>
        void serialize(Archive& ar) {
            ar(this->rows, this->cols);
        };
};

class Matrix
{
    private:
        int rows = 0;
        int cols = 0;
        std::vector<float> data;

    public:
        Matrix() = default;
        Matrix(const Shape& shape);
        Matrix(const Shape& shape, float defaultValue);
        Matrix(const Shape& shape, std::mt19937& rng, std::uniform_real_distribution<float>& defaultValueDistribution);
        Matrix(const std::vector<std::vector<float>>& mat);

        int rowCount() const;
        int colCount() const;
        bool empty() const;

        Shape shape() const;

        float get(int row, int col) const;
        void set(int row, int col, float value);

        int& dangerouslyGetRows();
        int& dangerouslyGetCols();
        std::vector<float>& dangerouslyGetData();

        std::string toString() const;

        template <class Archive>
        void serialize(Archive& ar) {
            ar(this->rows, this->cols, this->data);
        };

        static Matrix transpose(const Matrix& mat);
        static Matrix flipped(const Matrix& mat);
        static Matrix add(const Matrix& matA, const Matrix& matB);
        static Matrix subtract(const Matrix& matA, const Matrix& matB);
        static Matrix scalarProduct(const Matrix& mat, float scalar);
        static Matrix hadamardProduct(const Matrix& matA, const Matrix& matB);
        static Matrix matrixProduct(const Matrix& matA, const Matrix& matB);
        static Matrix matrixColumnProduct(const Matrix& mat, const Matrix& col);
        static Matrix upsample(const Matrix& mat, int gapSize);
};

#endif