#ifndef QCSNPE_H
#define QCSNPE_H

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <DlContainer/IDlContainer.hpp>
#include <DlSystem/RuntimeList.hpp>
#include <SNPE/SNPE.hpp>
#include <SNPE/SNPEBuilder.hpp>
//#include <DlSystem/UDLFunc.hpp>
#include <DlSystem/ITensorFactory.hpp>
#include <SNPE/SNPEFactory.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <android/log.h>
#include <chrono>
#include <ctime>

typedef std::chrono::milliseconds ms;

class Qcsnpe {
private:
    std::shared_ptr<zdl::SNPE::SNPE> model_handler;
    std::shared_ptr<zdl::DlContainer::IDlContainer> container;
    zdl::DlSystem::RuntimeList runtime_list;
    zdl::DlSystem::StringList outputs;
    zdl::DlSystem::TensorMap output_tensor_map;
    zdl::DlSystem::StringList out_tensors;

public:
    Qcsnpe(std::string &dlc, int system_type, std::vector<std::string> &output_layers);

    Qcsnpe(const Qcsnpe &qc);

    zdl::DlSystem::TensorMap predict(cv::Mat input_image);

    std::vector<float> throughput_vec;
    std::vector<float> fps_vec;
    bool keypoint_det_mode = true;
};

#endif
