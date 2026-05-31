#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "coreml_infer.hpp"

#include <p10_internal/log/log.hpp>
#include <ptensor/tensor.hpp>

namespace p10::infer {

struct CoreMLModel {
    MLModel* model;
    explicit CoreMLModel(MLModel* m) : model([m retain]) {}
    ~CoreMLModel() {
        [model release];
    }
    CoreMLModel(const CoreMLModel&) = delete;
    CoreMLModel& operator=(const CoreMLModel&) = delete;
};

namespace {

MLComputeUnits to_ml_compute_units(InferConfig::CoreMLComputeUnits units) {
    switch (units) {
        case InferConfig::CoreMLComputeUnits::CPUOnly:
            return MLComputeUnitsCPUOnly;
        case InferConfig::CoreMLComputeUnits::CPUAndGPU:
            return MLComputeUnitsCPUAndGPU;
        case InferConfig::CoreMLComputeUnits::CPUAndNeuralEngine:
            return MLComputeUnitsCPUAndNeuralEngine;
        case InferConfig::CoreMLComputeUnits::All:
            return MLComputeUnitsAll;
    }
    return MLComputeUnitsAll;
}

P10Result<MLMultiArrayDataType> dtype_to_ml(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float32:
            return Ok(MLMultiArrayDataTypeFloat32);
        case Dtype::Float64:
            return Ok(MLMultiArrayDataTypeDouble);
        case Dtype::Int32:
            return Ok(MLMultiArrayDataTypeInt32);
        default:
            return Err(P10Error::InvalidArgument << "Dtype not supported by CoreML MLMultiArray");
    }
}

P10Result<Dtype> ml_to_dtype(MLMultiArrayDataType ml_dtype) {
    switch (ml_dtype) {
        case MLMultiArrayDataTypeFloat32:
            return Ok(Dtype::Float32);
        case MLMultiArrayDataTypeDouble:
            return Ok(Dtype::Float64);
        case MLMultiArrayDataTypeInt32:
            return Ok(Dtype::Int32);
        default:
            return Err(P10Error::InvalidArgument << "Unsupported CoreML MLMultiArray data type");
    }
}

NSArray<NSNumber*>* shape_to_nsarray(std::span<const int64_t> shape) {
    NSMutableArray<NSNumber*>* arr = [NSMutableArray arrayWithCapacity:shape.size()];
    for (int64_t dim : shape) {
        [arr addObject:@(dim)];
    }
    return arr;
}

NSArray<NSNumber*>* row_major_strides(std::span<const int64_t> shape) {
    NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:shape.size()];
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        [strides insertObject:@(stride) atIndex:0];
        stride *= shape[i];
    }
    return strides;
}

}  // namespace

CoreMLInfer::CoreMLInfer(std::unique_ptr<CoreMLModel> model,
                         std::vector<std::string> input_names,
                         std::vector<std::string> output_names) :
    model_(std::move(model)),
    input_names_(std::move(input_names)),
    output_names_(std::move(output_names)) {}

CoreMLInfer::~CoreMLInfer() = default;

P10Result<std::unique_ptr<IInfer>> CoreMLInfer::create(const std::string& model_path,
                                                        const InferConfig& config) {
    @autoreleasepool {
        NSString* path_str = [NSString stringWithUTF8String:model_path.c_str()];
        NSURL* model_url = [NSURL fileURLWithPath:path_str];

        MLModelConfiguration* ml_config = [[MLModelConfiguration alloc] init];
        ml_config.computeUnits = to_ml_compute_units(config.coreml_compute_units());
        [ml_config autorelease];

        NSError* error = nil;
        MLModel* model = [MLModel modelWithContentsOfURL:model_url
                                           configuration:ml_config
                                                   error:&error];
        if (!model) {
            return Err(P10Error::InferError << error.localizedDescription.UTF8String);
        }

        MLModelDescription* desc = model.modelDescription;

        std::vector<std::string> input_names;
        for (NSString* name in desc.inputDescriptionsByName) {
            input_names.emplace_back(name.UTF8String);
        }

        std::vector<std::string> output_names;
        for (NSString* name in desc.outputDescriptionsByName) {
            output_names.emplace_back(name.UTF8String);
        }

        auto logger = p10::log::scope("CoreMLInfer");
        logger.info("CoreML model loaded: {}", model_path);
        logger.info("Number of inputs: {}", input_names.size());
        logger.info("Number of outputs: {}", output_names.size());

        return Ok<std::unique_ptr<IInfer>>(
            std::unique_ptr<IInfer>(new CoreMLInfer(std::make_unique<CoreMLModel>(model),
                                                    std::move(input_names),
                                                    std::move(output_names)))
        );
    }
}

P10Error CoreMLInfer::infer(std::span<Tensor> input_tensors, std::span<Tensor> output_tensors) {
    @autoreleasepool {
        if (input_tensors.size() != get_input_count()
            || output_tensors.size() != get_output_count()) {
            return P10Error::InvalidArgument << "Invalid number of input/output tensors";
        }

        NSError* error = nil;
        NSMutableDictionary<NSString*, MLFeatureValue*>* feature_dict =
            [NSMutableDictionary dictionaryWithCapacity:input_tensors.size()];

        for (size_t i = 0; i < input_tensors.size(); ++i) {
            Tensor& tensor = input_tensors[i];
            const auto shape_span = tensor.shape().as_span();

            auto ml_dtype_result = dtype_to_ml(tensor.dtype());
            if (!ml_dtype_result.is_ok()) {
                return ml_dtype_result.error();
            }

            MLMultiArray* ml_array = [[MLMultiArray alloc]
                initWithDataPointer:tensor.as_bytes().data()
                              shape:shape_to_nsarray(shape_span)
                           dataType:ml_dtype_result.unwrap()
                            strides:row_major_strides(shape_span)
                        deallocator:nil
                              error:&error];
            [ml_array autorelease];
            if (!ml_array) {
                return P10Error::InferError << error.localizedDescription.UTF8String;
            }

            NSString* name = [NSString stringWithUTF8String:input_names_[i].c_str()];
            feature_dict[name] = [MLFeatureValue featureValueWithMultiArray:ml_array];
        }

        MLDictionaryFeatureProvider* input_features =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:feature_dict error:&error];
        [input_features autorelease];
        if (!input_features) {
            return P10Error::InferError << error.localizedDescription.UTF8String;
        }

        id<MLFeatureProvider> output_features =
            [model_->model predictionFromFeatures:input_features error:&error];
        if (!output_features) {
            return P10Error::InferError << error.localizedDescription.UTF8String;
        }

        for (size_t i = 0; i < output_tensors.size(); ++i) {
            NSString* name = [NSString stringWithUTF8String:output_names_[i].c_str()];
            MLFeatureValue* feature_value = [output_features featureValueForName:name];
            if (!feature_value) {
                return P10Error::InferError
                    << ("Missing output feature: " + output_names_[i]);
            }
            if (feature_value.type != MLFeatureTypeMultiArray) {
                return P10Error::InferError << "Output feature is not a MultiArray";
            }

            MLMultiArray* ml_output = feature_value.multiArrayValue;

            auto dtype_result = ml_to_dtype(ml_output.dataType);
            if (!dtype_result.is_ok()) {
                return dtype_result.error();
            }

            std::vector<int64_t> shape_vec;
            shape_vec.reserve(ml_output.shape.count);
            for (NSNumber* dim in ml_output.shape) {
                shape_vec.push_back(dim.longLongValue);
            }

            auto shape_result = make_shape(std::span<const int64_t>(shape_vec));
            if (!shape_result.is_ok()) {
                return shape_result.error();
            }

            P10_RETURN_IF_ERROR(
                output_tensors[i].create(shape_result.unwrap(),
                                         TensorOptions().dtype(dtype_result.unwrap()))
            );
            std::memcpy(output_tensors[i].as_bytes().data(),
                        ml_output.dataPointer,
                        output_tensors[i].size_bytes());
        }

        return P10Error::Ok;
    }
}

}  // namespace p10::infer
