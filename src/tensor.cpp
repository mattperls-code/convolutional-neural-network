#include "tensor.hpp"

Tensor::Tensor(const Dimensions& dimensions)
{
    this->data = std::vector<Matrix>(dimensions.depth, Matrix(dimensions.shape));
};

Tensor::Tensor(const Dimensions& dimensions, float defaultValue)
{
    this->data = std::vector<Matrix>(dimensions.depth, Matrix(dimensions.shape, defaultValue));
};

Tensor::Tensor(const Dimensions& dimensions, std::mt19937& rng, std::uniform_real_distribution<float>& defaultValueDistribution)
{
    this->data = std::vector<Matrix>(dimensions.depth, Matrix(dimensions.shape, rng, defaultValueDistribution));
};

Tensor::Tensor(const std::vector<Matrix>& tensor)
{
    this->data = tensor;
};

Dimensions Tensor::getDimensions() const
{
    if (this->data.empty()) throw std::runtime_error("Tensor getDimensions: tensor is empty");

    return Dimensions(this->data.size(), this->data[0].shape());
};

bool Tensor::empty() const
{
    return this->data.empty() || this->data[0].empty();
};

float Tensor::get(int row, int col, int depth) const
{
    if (depth < 0 || depth >= this->data.size()) throw std::runtime_error("Tensor get: depth out of bounds");

    return this->data[depth].get(row, col);
};

void Tensor::set(int row, int col, int depth, float value)
{
    if (depth < 0 || depth >= this->data.size()) throw std::runtime_error("Tensor set: depth out of bounds");
    
    this->data[depth].set(row, col, value);
};

std::vector<Matrix>& Tensor::dangerouslyGetData()
{
    return this->data;
};

std::string Tensor::toString() const
{
    if (this->empty()) return "{ }";

    std::string output;

    output += "{ ";

    for (auto matrix : this->data) output += matrix.toString() + ", ";

    output.pop_back();
    output.pop_back();

    output += " }";

    return output;
};