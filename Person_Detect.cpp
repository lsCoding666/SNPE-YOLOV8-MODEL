#include "Person_Detect.h"
#include <unistd.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <string>
#include <cstdlib>
#include <mutex>
#include <glob.h>
#include <dirent.h>
#include <stdio.h>
#include <opencv2/imgproc/types_c.h>


Person_Detect::Person_Detect()
        : m_camera_ready(false), m_image(nullptr), m_image_reader(nullptr) {}

Person_Detect::~Person_Detect() {
    JNIEnv *env;
    java_vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6);
    env->DeleteGlobalRef(calling_activity_obj);
    calling_activity_obj = nullptr;

    // ACameraCaptureSession_stopRepeating(m_capture_session);

    // make sure we don't leak native windows
    if (m_native_window != nullptr) {
        ANativeWindow_release(m_native_window);
        m_native_window = nullptr;
    }

    if (m_image_reader != nullptr) {
        delete (m_image_reader);
        m_image_reader = nullptr;
    }
}

void Person_Detect::OnCreate() {
    //0 cpu 1gpu 2dsp
    qc = new Qcsnpe(model_path, 1, output_layers);

}

void Person_Detect::OnPause() {}

void Person_Detect::OnDestroy() {}

void Person_Detect::SetNativeWindow(ANativeWindow *native_window) {
    // Save native window
    m_native_window = native_window;
}


//std::string class_name_path = "/storage/emulated/0/Documents/classes_traffic2.txt";
//std::string class_name_path = "/storage/emulated/0/Documents/classes_traffic.txt";
std::string class_name_path = "/storage/emulated/0/Documents/classes.txt";

std::vector<std::string> load_class_list() {
    std::vector<std::string> class_list;
    std::ifstream ifs(class_name_path);
    std::string line;
    while (getline(ifs, line)) {
        class_list.push_back(line);
    }
    return class_list;
}

std::vector<std::string> class_list = load_class_list();

//
//void Person_Detect::CameraLoop() {
//    bool buffer_printout = false;
//    //video_writer.open("/sdcard/Documents/Person_Detect_video.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10.0, cv::Size(640, 480), true);
//
//    while (1) {
//        if (m_camera_thread_stopped) { break; }
//        if (!m_camera_ready || !m_image_reader) { continue; }
//        //reading the image from ndk reader
//        m_image = m_image_reader->GetNextImage();
//        if (m_image == nullptr) { continue; }
//
//        ANativeWindow_acquire(m_native_window);
//        ANativeWindow_Buffer buffer;
//        if (ANativeWindow_lock(m_native_window, &buffer, nullptr) < 0) {
//            m_image_reader->DeleteImage(m_image);
//            m_image = nullptr;
//            continue;
//        }
//        if (false == buffer_printout) {
//            buffer_printout = true;
//            LOGI("/// H-W-S-F: %d, %d, %d, %d", buffer.height, buffer.width, buffer.stride,
//                 buffer.format);
//        }
//
//        //display the image
//        m_image_reader->DisplayImage(&buffer, m_image);
//
//        //converting the ndk image into opencv format
//        img_mat = cv::Mat(buffer.height, buffer.stride, CV_8UC4, buffer.bits);
//        //cv::imwrite("/storage/emulated/0/appData/models/input.jpg",img_mat);
//        cv::Mat src_img = img_mat.clone();
//
//        bgr_img = cv::Mat(img_mat.rows, img_mat.cols, CV_8UC3);
//
//        cv::cvtColor(img_mat, bgr_img, cv::COLOR_RGBA2BGR);
//        // bgr_img is normal image
//        //cv::imwrite("/storage/emulated/0/appData/models/inp.jpg",bgr_img);
//
//        cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
//        std::vector<Detection> output;
//        cv::Mat res_img = cv::Mat(640, 640, CV_8UC3);
//        cv::resize(bgr_img, res_img, cv::Size(640, 640));
//        // res_img is pre-processed image we are passing it for inference
//        //cv::imwrite("/storage/emulated/0/appData/models/inp.jpg",res_img);
//
//        pred_out = qc->predict(res_img);
//
//
//        std::vector<float> out_arr = pred_out["output"];
//        std::vector<cv::Mat> outputs;
//
//        outputs.emplace_back(cv::Mat(out_arr));
//        float x_factor = res_img.cols / INPUT_WIDTH;
//        float y_factor = res_img.rows / INPUT_HEIGHT;
//
//        //float *data = (float *)outputs[0].data;
//        float *data = (float *) out_arr.data();
//        //const int dimensions = 85;
//        const int dimensions = 6;
//        const int rows = 25200;
//
//        std::vector<int> class_ids;
//        std::vector<float> confidences;
//        std::vector<cv::Rect> boxes;
//
//
//        for (int i = 0; i < rows; ++i) {
//
//            float confidence = data[4];
//            if (confidence >= CONFIDENCE_THRESHOLD) {
//
//                float *classes_scores = data + 5;
//                cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
//                cv::Point class_id;
//                double max_class_score;
//                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
//                if (max_class_score > SCORE_THRESHOLD) {
//
//                    confidences.push_back(confidence);
//
//                    class_ids.push_back(class_id.x);
//
//                    float x = data[0];
//                    float y = data[1];
//                    float w = data[2];
//                    float h = data[3];
//                    int left = int((x - 0.5 * w) * x_factor);
//                    int top = int((y - 0.5 * h) * y_factor);
//                    int width = int(w * x_factor);
//                    int height = int(h * y_factor);
//                    boxes.push_back(cv::Rect(left, top, width, height));
//                }
//
//            }
//            data += 85;
//
//        }
//
//        //std::cout << class_ids.size() << std::endl;
//
//        std::vector<int> nms_result;
//        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
//        for (int i = 0; i < nms_result.size(); i++) {
//            int idx = nms_result[i];
//            Detection result;
//            result.class_id = class_ids[idx];
//            __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "tag %d", class_ids[idx]);
//            result.confidence = confidences[idx];
//            result.box = boxes[idx];
//            output.push_back(result);
//        }
//        int detections = output.size();
//
//        LOGI("%d", detections);
//        for (int i = 0; i < detections; ++i) {
//
//            auto detection = output[i];
//
//            auto box = detection.box;
//            auto classId = detection.class_id;
//            const auto color = colors[classId % colors.size()];
//            cv::rectangle(img_mat, box, color, 3);
//            cv::rectangle(img_mat, cv::Point(box.x, box.y - 20),
//                          cv::Point(box.x + box.width, box.y), color, cv::FILLED);
//        }
//        cv::imwrite("/storage/emulated/0/appData/models/Person_Detect_bgr.jpg", bgr_img);
//        cv::resize(img_mat, out_img, cv::Size(640, 480));
//        video_writer.write(out_img);
//        cv::imwrite("/storage/emulated/0/appData/models/Person_Detect_image.jpg", out_img);
//
//        pred_out.clear();
//        ANativeWindow_unlockAndPost(m_native_window);
//        ANativeWindow_release(m_native_window);
//    }
//    video_writer.release();
//
//}
cv::Mat Person_Detect::ProcessImgYoloV8(cv::Mat mat, char *pJstring) {
    img_mat = mat;

    std::vector<Detection> output;
    cv::Mat res_img = cv::Mat(640, 640, CV_8UC3);

    cv::Mat input_mat;
    im_scale = std::min((float) INPUT_WIDTH / img_mat.cols, (float) INPUT_HEIGHT / img_mat.rows);

    int new_w = int(img_mat.cols * im_scale);
    int new_h = int(img_mat.rows * im_scale);
    cv::resize(img_mat, input_mat, cv::Size(new_w, new_h));    //resize

    int p_w = INPUT_WIDTH - new_w;
    int p_h = INPUT_WIDTH - new_h;

    int top = p_h / 2;
    int bottom = p_h - top;

    int left = p_w / 2;
    int right = p_w - left;

    cv::copyMakeBorder(input_mat, input_mat,        //原图像与扩充后的图像
                       top, bottom,                 //表示在图像四周扩充边缘的大小，top,bottom,left,right
                       left, right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    //开始预测
    zdl::DlSystem::TensorMap output_tensor_map = qc->predict(input_mat);
    zdl::DlSystem::StringList out_tensors = output_tensor_map.getTensorNames();


    out_tensors = output_tensor_map.getTensorNames();
    std::map<std::string, std::vector<float>> out_itensor_map;
    for (size_t i = 0; i < out_tensors.size(); i++) {
        zdl::DlSystem::ITensor *out_itensor = output_tensor_map.getTensor(out_tensors.at(i));
        std::vector<float> out_vec{reinterpret_cast<float *>(&(*out_itensor->begin())),
                                   reinterpret_cast<float *>(&(*out_itensor->end()))};
        out_itensor_map.insert(std::make_pair(std::string(out_tensors.at(i)), out_vec));
    }


    std::vector<BoxInfo> result;
    zdl::DlSystem::ITensor *out_itensor = output_tensor_map.getTensor(out_tensors.at(0));
    auto boxes = Person_Detect::decode_inferV8(out_itensor->begin().dataPointer(),
                                               {(int) img_mat.cols, (int) img_mat.rows},
                                               left, top,
                                               class_list.size(),
                                               CONFIDENCE_THRESHOLD);
    result.insert(result.begin(), boxes.begin(), boxes.end());


    Person_Detect::nms(result, NMS_THRESHOLD);
    for (int i = 0; i < result.size(); ++i) {

        auto detection = result[i];
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "tag %d", detection.label);
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "tag %f", detection.score);
        cv::Scalar color = cv::Scalar(255, 255, 0);
        cv::rectangle(img_mat, cv::Point(detection.x1, detection.y1),
                      cv::Point(detection.x2, detection.y2),
                      color,2);
        cv::rectangle(img_mat, cv::Point(detection.x1, detection.y1 - 20), cv::Point(detection.x2, detection.y1 ),
                      color,-1);
        std::stringstream ss;
        ss << class_list[detection.label]  << detection.score;
        cv::putText(img_mat, ss.str(), cv::Point(detection.x1, detection.y1),
                    cv::FONT_HERSHEY_COMPLEX, 0.8,
                    cv::Scalar(0, 0, 0), 2);
    }
    std::string str1 = "/storage/emulated/0/testresult/";
    std::string str2 = ".jpg";
    cvtColor(img_mat, img_mat, CV_RGB2BGR);
    cv::imwrite(str1.append(pJstring).append(str2), img_mat);
    pred_out.clear();
    return img_mat;
}

cv::Mat Person_Detect::ProcessImg(cv::Mat mat, char *pJstring) {
    img_mat = mat;

    std::vector<Detection> output;
    cv::Mat res_img = cv::Mat(640, 640, CV_8UC3);

    cv::Mat input_mat;
    im_scale = std::min((float) INPUT_WIDTH / img_mat.cols, (float) INPUT_HEIGHT / img_mat.rows);

    int new_w = int(img_mat.cols * im_scale);
    int new_h = int(img_mat.rows * im_scale);
    cv::resize(img_mat, input_mat, cv::Size(new_w, new_h));    //resize

    int p_w = INPUT_WIDTH - new_w;
    int p_h = INPUT_WIDTH - new_h;

    int top = p_h / 2;
    int bottom = p_h - top;

    int left = p_w / 2;
    int right = p_w - left;

    cv::copyMakeBorder(input_mat, input_mat,        //原图像与扩充后的图像
                       top, bottom,                 //表示在图像四周扩充边缘的大小，top,bottom,left,right
                       left, right,
                       cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));

    //开始预测
    zdl::DlSystem::TensorMap output_tensor_map = qc->predict(input_mat);
    zdl::DlSystem::StringList out_tensors = output_tensor_map.getTensorNames();


//    out_tensors = output_tensor_map.getTensorNames();
//    std::map<std::string, std::vector<float>> out_itensor_map;
//    for (size_t i = 0; i < out_tensors.size(); i++) {
//        zdl::DlSystem::ITensor *out_itensor = output_tensor_map.getTensor(out_tensors.at(i));
//        std::vector<float> out_vec{reinterpret_cast<float *>(&(*out_itensor->begin())),
//                                   reinterpret_cast<float *>(&(*out_itensor->end()))};
//        out_itensor_map.insert(std::make_pair(std::string(out_tensors.at(i)), out_vec));
//    }

    std::vector<BoxInfo> result;
    for (int i = 0; i < out_tensors.size(); ++i) {
        zdl::DlSystem::ITensor *out_itensor = output_tensor_map.getTensor(out_tensors.at(i));
        //输出头的名字
        std::string name = std::string(out_tensors.at(i));
        for (const auto &item: layers) {
            if (item.index == name) {
                auto boxes = Person_Detect::decode_infer(out_itensor->begin().dataPointer(),
                                                         item.stride,
                                                         {(int) img_mat.cols, (int) img_mat.rows},
                                                         left, top,
                                                         INPUT_WIDTH,
                                                         class_list.size(), item.anchors,
                                                         CONFIDENCE_THRESHOLD,
                                                         item.grid_size);
                result.insert(result.begin(), boxes.begin(), boxes.end());
            }
        }
    }


    Person_Detect::nms(result, NMS_THRESHOLD);
    for (int i = 0; i < result.size(); ++i) {

        auto detection = result[i];
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "tag %d", detection.label);
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "tag %f", detection.score);
        cv::Scalar color = cv::Scalar(255, 255, 0);
        cv::rectangle(img_mat, cv::Point(detection.x1, detection.y1),
                      cv::Point(detection.x2, detection.y2),
                      color,2);
        cv::rectangle(img_mat, cv::Point(detection.x1, detection.y1 - 20), cv::Point(detection.x2, detection.y1 ),
                      color,
                      2, cv::FILLED);
        std::stringstream ss;
        ss << class_list[detection.label] << detection.score;
        cv::putText(img_mat, ss.str(), cv::Point(detection.x1, detection.y1),
                    cv::FONT_HERSHEY_COMPLEX, 0.8,
                    color, 2);
    }
    std::string str1 = "/storage/emulated/0/testresult/";
    std::string str2 = ".jpg";
    cvtColor(img_mat, img_mat, CV_RGB2BGR);
    cv::imwrite(str1.append(pJstring).append(str2), img_mat);
    pred_out.clear();
    return img_mat;
}


inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<BoxInfo>
Person_Detect::decode_inferV8(float *dataSource, const YoloSize &frame_size,
                              int left, int top,
                              int num_classes, float threshold) {
    float *data = dataSource;
    std::vector<BoxInfo> result;
//    float cx, cy, w, h;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    for (int i = 0; i < 8400; ++i) {
//        float *classes_scores = data + 4;
//        cv::Mat scores(1, num_classes, CV_32FC1, classes_scores);
//        cv::Point class_id;
//        double maxClassScore;
        std::vector<int> class_ids;
//        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        float maxScore = 0;
        int maxClass = -1;
        for (int cls = 0; cls < num_classes; cls++) {
            float score =
                    data[cls + 4];
            if (score > maxScore) {
                maxScore = score;
                maxClass = cls;
            }
        }
        if (i == 7255){
            int a = 0;
        }
        if (maxScore > threshold) {
            confidences.push_back(maxScore);
            class_ids.push_back(maxClass);
            BoxInfo box;
            float w = data[2];
            float h = data[3];

            box.x1 = std::max(0, std::min(frame_size.width,
                                          int((data[0] - w / 2.f - left) / im_scale)));
            box.y1 = std::max(0, std::min(frame_size.height,
                                          int((data[1] - h / 2.f - top) / im_scale)));
            box.x2 = std::max(0, std::min(frame_size.width,
                                          int((data[0] + w / 2.f - left) / im_scale)));
            box.y2 = std::max(0, std::min(frame_size.height,
                                          int((data[1] + h / 2.f - top) / im_scale)));
            box.score = maxScore;
            box.label = maxClass;

            result.push_back(box);
        }
        data += 84;
    }
    return result;
}

std::vector<BoxInfo>
Person_Detect::decode_infer(float *dataSource, int stride, const YoloSize &frame_size,
                            int left, int top,
                            int net_size,
                            int num_classes,
                            const std::vector<YoloSize> &anchors, float threshold, int grid_size) {
    float *data = dataSource;
    std::vector<BoxInfo> result;
    float cx, cy, w, h;
    for (int shift_y = 0; shift_y < grid_size; shift_y++) {
        for (int shift_x = 0; shift_x < grid_size; shift_x++) {
            int loc = shift_x + shift_y * grid_size;
            //一个头有三个anchors
            for (int i = 0; i < 3; i++) {
//                int index = i * 85 + loc;
                float maxScore = 0;
                int maxClass = -1;
                for (int cls = 0; cls < num_classes; cls++) {
                    float score = data[4] *
                                  data[cls + 5];
                    if (score > maxScore) {
                        maxScore = score;
                        maxClass = cls;
                    }
                }
                if (maxScore > threshold) {
                    cx = (data[0] * 2.f - 0.5f + (float) shift_x) * (float) stride;
                    cy = (data[1] * 2.f - 0.5f + (float) shift_y) * (float) stride;
                    w = pow(data[2] * 2.f, 2) * anchors[i].width;
                    h = pow(data[3] * 2.f, 2) * anchors[i].height;
                    //printf("[grid size=%d, stride = %d]x y w h %f %f %f %f\n",grid_size,stride,record[0],record[1],record[2],record[3]);
                    BoxInfo box;
                    box.x1 = std::max(0, std::min(frame_size.width,
                                                  int((cx - w / 2.f - left) / im_scale)));
                    box.y1 = std::max(0, std::min(frame_size.height,
                                                  int((cy - h / 2.f - top) / im_scale)));
                    box.x2 = std::max(0, std::min(frame_size.width,
                                                  int((cx + w / 2.f - left) / im_scale)));
                    box.y2 = std::max(0, std::min(frame_size.height,
                                                  int((cy + h / 2.f - top) / im_scale)));
                    box.score = maxScore;
                    box.label = maxClass;
                    result.push_back(box);

                }
                data = data + num_classes + 5;
            }
        }
    }
    return result;
}

void Person_Detect::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}

