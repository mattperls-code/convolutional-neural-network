#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "matrix.hpp"

#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

class Dimensions
{
    public:
        int depth = 0;
        Shape shape;

        Dimensions() = default;
        Dimensions(int depth, const Shape& shape): depth(depth), shape(shape) {};

        bool operator==(const Dimensions&) const = default;

        int volume() const;

        std::string toString() const;

        template <class Archive>
        void serialize(Archive& ar) {
            ar(this->depth, this->shape);
        };
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
        
        Matrix getMatrix(int depth) const;
        Matrix getColumnVector() const;

        std::string toString() const;

        template <class Archive>
        void serialize(Archive& ar) {
            ar(this->data);
        };

        static Tensor fromColumnVector(Matrix& columnVector, const Dimensions& tensorDimensions);

        static Tensor add(const Tensor& tensorA, const Tensor& tensorB);
        static Tensor scalarProduct(const Tensor& tensor, float scalar);
};

#endif