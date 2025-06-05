#include "tensor.hpp"

int Dimensions::volume() const
{
    return this->depth * this->shape.area();
};

std::string Dimensions::toString() const
{
    return this->shape.toString() + " by " + std::to_string(this->depth);
};

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

Matrix Tensor::getMatrix(int depth) const
{
    if (depth < 0 || depth >= this->data.size()) throw std::runtime_error("Tensor getMatrix: depth out of bounds");

    return this->data[depth];
};

Matrix Tensor::getColumnVector() const
{
    Matrix output(Shape(this->getDimensions().volume(), 1));

    auto& outputData = output.dangerouslyGetData();

    int offset = 0;

    for (auto matrix : this->data) {
        for (auto value : matrix.dangerouslyGetData()) {
            outputData[offset] = value;

            offset++;
        }
    }

    return output;
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

Tensor Tensor::fromColumnVector(Matrix& columnVector, const Dimensions& tensorDimensions)
{
    std::vector<Matrix> tensor(tensorDimensions.depth, Matrix(tensorDimensions.shape));

    for (int i = 0;i<tensor.size();i++) tensor[i].dangerouslyGetData() = std::vector<float>(
        columnVector.dangerouslyGetData().begin() + (i * tensorDimensions.shape.area()),
        columnVector.dangerouslyGetData().begin() + ((i + 1) * tensorDimensions.shape.area())
    );

    return Tensor(tensor);
};

Tensor Tensor::add(const Tensor& tensorA, const Tensor& tensorB)
{
    if (tensorA.getDimensions() != tensorB.getDimensions()) throw std::runtime_error("Tensor add: inputs are different dimensions");

    Tensor output(tensorA.getDimensions());
    
    for (int i = 0;i<output.data.size();i++) output.data[i] = Matrix::add(tensorA.data[i], tensorB.data[i]);

    return output;
};

Tensor Tensor::scalarProduct(const Tensor& tensor, float scalar)
{
    Tensor output = tensor;

    for (auto& matrix : output.data) matrix = Matrix::scalarProduct(matrix, scalar);

    return output;
};