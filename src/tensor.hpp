#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "../lib/matrix.hpp"

class Dimensions
{
    public:
        int depth;
        Shape shape;

        Dimensions(int depth, const Shape& shape): depth(depth), shape(shape) {};

        bool operator==(const Dimensions&) const = default;
};

class Tensor
{
    private:
        std::vector<Matrix> data;

    public:
        Tensor() = default;
        Tensor(const Dimensions& dimensions);
        Tensor(const Dimensions& dimensions, float defaultValue);
        Tensor(const Dimensions& dimensions, std::mt19937& rng, std::uniform_real_distribution<float>& defaultValueDistribution);
        Tensor(const std::vector<Matrix>& tensor);

        Dimensions getDimensions() const;

        bool empty() const;

        float get(int row, int col, int depth) const;
        void set(int row, int col, int depth, float value);

        std::vector<Matrix>& dangerouslyGetData();

        std::string toString() const;
};

#endif