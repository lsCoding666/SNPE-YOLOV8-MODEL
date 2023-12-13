#include <cstdio>
#include <android/log.h>
#include <android/native_window.h>
#include <android_native_app_glue.h>
#include <functional>
#include <thread>
#include "qcsnpe.hpp"
#include "Util.h"
#include <stdlib.h>

/************************************************************************
* Name : Qcsnpe <Constructor>
* Function: Constructor checks runtime availability, set runtime, set
*           output layers and then loads the DLC model to model handler 
************************************************************************/
Qcsnpe::Qcsnpe(std::string &dlc, int system_type, std::vector<std::string> &output_layers) {
    std::ifstream dlc_file(dlc);
    zdl::DlSystem::Runtime_t runtime_cpu = zdl::DlSystem::Runtime_t::CPU;
    zdl::DlSystem::Runtime_t runtime_gpu = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
    zdl::DlSystem::Runtime_t runtime_dsp = zdl::DlSystem::Runtime_t::DSP;
    zdl::DlSystem::Runtime_t runtime_aip = zdl::DlSystem::Runtime_t::AIP_FIXED8_TF;
    zdl::DlSystem::PerformanceProfile_t perf = zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE;

    if (!dlc_file) {
        LOGI("%s\n",
             "Dlc file not valid. Please ensure that you have provided a valid dlc for processing.");
        exit(0);
    } else {
        LOGI("%s\n",
             "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Dlc file created>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    }

    //Loading Model and setting Runtime
    static zdl::DlSystem::Version_t Version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    LOGI("%s %s %s\n", "<<<<<<<<<<<<<<<<<<<SNPE Version: ", Version.asString().c_str(),
         ">>>>>>>>>>>>>>>>>>>>>>>>>>>");
    container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlc.c_str()));
    if (container == nullptr) {
        LOGI("%s\n",
             "<<<<<<<<<<<<<<<<<<<<<<<Error while opening the container file.>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    }

    switch (system_type) {
        case 3:
            if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime_aip)) {
                runtime_list.add(runtime_aip);
                LOGI("%s\n",
                     "<<<<<<<<<<<<<<<<<<<<<<<<AIP added to runtime list>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
            } else
                LOGI("%s\n",
                     "<<<<<<<<<<<<<<<<<<<<<<<AIP not available>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
        case 2:
            if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime_dsp)) {
                runtime_list.add(runtime_dsp);
                LOGI("%s\n",
                     "<<<<<<<<<<<<<<<<<<<DSP added to runtime list>>>>>>>>>>>>>>>>>>>>>>>>");
            } else
                LOGI("%s\n", "<<<<<<<<<<<<<<<<<<<<<<DSP not available>>>>>>>>>>>>>>>>>>>>>>");
        case 1:
            if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime_gpu)) {
                runtime_list.add(runtime_gpu);
                LOGI("%s\n", "<<<<<<<<<<<<<<<<<<<GPU added to runtime list>>>>>>>>>>>>>>>>>>>>>>>");
            } else
                LOGI("%s\n", "<<<<<<<<<<<<<<<<<<GPU not available>>>>>>>>>>>>>>>>>>>>");
//                break;
        case 0:
            if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime_cpu)) {
                runtime_list.add(runtime_cpu);
                LOGI("%s\n", "<<<<<<<<<<<<<<<<CPU added to runtime list>>>>>>>>>>>>>>>>>>>");
            } else
                LOGI("%s\n", "<<<<<<<<<<<<<<<<<<<<<<<<<<<CPU not available>>>>>>>>>>>>>>>>>>>>>");
            break;
        default:
            LOGI("%s\n",
                 "<<<<<<<<<<<<<<<<<<<<<<<<<Runtime invalid. Setting to CPU>>>>>>>>>>>>>>>>>>>>>>>>>");
            runtime_list.add(runtime_cpu);
            break;
    }
    if (runtime_list.size() > 1) {
        LOGI("%s\n",
             "<<<<<<<<<<<<<<<<<<<<<<<<<Multiple runtime available. Fallback enabled>>>>>>>>>>>>>>>>>>>>>>>>>");
    }
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    LOGI("%s\n", "<<<<<<<<<<<<<<<<<<<<<< Done snpeBuilder 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>");
    for (auto &output_layer: output_layers) {
        outputs.append(output_layer.c_str());
    }
    LOGI("%s\n", "<<<<<<<<<<<<<<< Done snpeBuilder 2 >>>>>>>>>>>>>>>>");
    zdl::DlSystem::PlatformConfig platform_config;
//    platform_config.setPlatformOptions("unsignedPD:ON");
    model_handler = snpeBuilder.setOutputLayers(outputs)
            .setRuntimeProcessorOrder(runtime_list)
            .setPlatformConfig(platform_config)
            .build();
    LOGI("%s\n", "<<<<<<<<<<<<<<<<<< Done snpeBuilder 3 >>>>>>>>>>>>>>>>>>");
    if (model_handler == nullptr) {
        LOGI("%s\n",
             "<<<<<<<<<<<<<<<<<<<< Error during creation of SNPE object. >>>>>>>>>>>>>>>>>>>>>>>");
    }
    LOGI("%s\n", "<<<<<<<<<<<<<<<<<<<<<<< Done snpeBuilder 4 >>>>>>>>>>>>>>>>>>>>>>>>>>");
}

Qcsnpe::Qcsnpe(const Qcsnpe &qc) {
    model_handler = std::move(qc.model_handler);
    container = std::move(qc.container);
    runtime_list = qc.runtime_list;
    outputs = qc.outputs;
    output_tensor_map = qc.output_tensor_map;
    out_tensors = qc.out_tensors;
}

/************************************************************************
* Name : predict
* Function: Method of qcsnpe class for infrencing
* Returns: A STL map with output tensor as key and its corresponding
*          output as value of map.
************************************************************************/
zdl::DlSystem::TensorMap Qcsnpe::predict(cv::Mat input_image) {
    unsigned long int in_size = 1;
    const zdl::DlSystem::TensorShape i_tensor_shape = model_handler->getInputDimensions();
    const zdl::DlSystem::Dimension *shapes = i_tensor_shape.getDimensions();
    int img_size = input_image.channels() * input_image.cols * input_image.rows;
    for (int i = 1; i < i_tensor_shape.rank(); i++) {
        in_size *= shapes[i];
    }

    if (in_size != img_size) {
        LOGI("%s\n", "Input Size mismatch!");
        LOGI("%s\n", "Expected: ");
        LOGI("%d\n", img_size);
        LOGI("%s\n", "Got: ");
        LOGI("%lu\n", in_size);
    }
    //获得模型输入头
    std::unique_ptr<zdl::DlSystem::ITensor> input_tensor =
            zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(
                    model_handler->getInputDimensions());
    zdl::DlSystem::ITensor *tensor_ptr = input_tensor.get();
    if (tensor_ptr == nullptr) {
        LOGI("%s\n", "Could not create SNPE input tensor");
    }
    // 获得模型输入头起点，准备填充数据
    float *tensor_ptr_fl = reinterpret_cast<float *>(&(*input_tensor->begin()));

// 使用指针访问像素，并进行类型转换和归一化
    const int channels = input_image.channels();
    const int rows = input_image.rows;
    const int cols = input_image.cols;

    for (int i = 0; i < rows; i++) {
        const uchar *row_ptr = input_image.ptr<uchar>(i);
        for (int j = 0; j < cols; j++) {
            const uchar *pixel_ptr = row_ptr + j * channels;
            for (int k = 0; k < channels; k++) {
                tensor_ptr_fl[(i * cols + j) * channels + k] =
                        static_cast<float>(pixel_ptr[k])/255;
            }
        }
    }

    //infer
    bool exec_status = model_handler->execute(tensor_ptr, output_tensor_map);
    if (!exec_status) {
//        LOGI("%s\n", "Error while executing the network.");
    }

    return output_tensor_map;
}

//    throughput_vec.push_back(d.count());
//    fps_vec.push_back(1.0/elapsed_time.count());



