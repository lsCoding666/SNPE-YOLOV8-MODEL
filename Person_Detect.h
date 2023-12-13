
#ifndef PERSONDETECTION_CAMERAAPP_H
#define PERSONDETECTION_CAMERAAPP_H


#include <android/asset_manager.h>
#include <android/native_window.h>
#include <jni.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/features2d.hpp>

#include "Image_Reader.h"
#include "Util.h"

#include <unistd.h>
#include <time.h>

#include <cstdlib>
#include <string>
#include <vector>
#include <thread>
#include "qcsnpe.hpp"
//#include "utils.hpp"
#define OUTPUT_LAYER_1 "/model.24/Concat_15"

typedef struct {
    int width;
    int height;
} YoloSize;

typedef struct BoxInfo {
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int label;
} BoxInfo;

typedef struct {
    std::string index;
    int stride;
    std::vector<YoloSize> anchors;
    int grid_size;
} YoloLayerData;

class Person_Detect {
public:
    Person_Detect();

    ~Person_Detect();

    Person_Detect(const Person_Detect &other) = delete;

    Person_Detect &operator=(const Person_Detect &other) = delete;

    void OnCreate();

    void OnPause();

    void OnDestroy();

    void SetJavaVM(JavaVM *pjava_vm) { java_vm = pjava_vm; }

    void SetNativeWindow(ANativeWindow *native_window);

    void SetUpCamera();

    void CameraLoop();

    cv::Mat ProcessImg(cv::Mat mat, char *pJstring);

    std::vector<BoxInfo>
    decode_infer(float *dataSource, int stride, const YoloSize &frame_size,
                 int left, int top,
                 int net_size,
                 int num_classes,
                 const std::vector<YoloSize> &anchors, float threshold, int grid_size);

    void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);

    cv::Mat ProcessImgYoloV8(cv::Mat mat, char *pJstring);

    std::vector<BoxInfo>decode_inferV8(float *dataSource, const YoloSize &frame_size,
                                  int left, int top,
                                  int num_classes, float threshold);


private:
    JavaVM *java_vm;
    jobject calling_activity_obj;
    ANativeWindow *m_native_window;
    ANativeWindow_Buffer m_native_buffer;
    ImageFormat m_view{0, 0, 0};
    Image_Reader *m_image_reader;
    AImage *m_image;

    volatile bool m_camera_ready;
    // for timing OpenCV bottlenecks
    clock_t start_t, end_t;
    // Used to detect up and down motion
    bool scan_mode;

    // OpenCV values
    cv::Mat img_mat;
    cv::Mat bgr_img;
    cv::Mat grey_img;
    cv::Mat rgb_img;

    cv::Mat out_img;
    bool m_camera_thread_stopped = false;
    Qcsnpe *qc;


    cv::VideoWriter video_writer;

//    std::string model_path = "/storage/emulated/0/appData/models/yolov5_person_latest.dlc";
//    std::string model_path = "/sdcard/Download/Telegram/yolov5_person_latest.dlc";
//    std::string model_path = "/data/local/tmp/incpv3/yolov5s_3head.dlc";
//    std::string model_path = "/data/local/tmp/incpv3/road_facilities_s.dlc";
    std::string model_path = "/data/local/tmp/incpv3/yolov8m_htp.dlc";
    std::vector<std::string> output_layers{"Transpose_325"};
//    std::vector<std::string> output_layers{"Sigmoid_199", "Sigmoid_201", "Sigmoid_203"};
//    std::vector<std::string> output_layers{"Sigmoid_272", "Sigmoid_274", "Sigmoid_276"};
    std::vector<std::vector<float>> pred_out;

    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
//    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.5;
    const float CONFIDENCE_THRESHOLD = 0.3;

    struct Detection {
        int class_id;
        float confidence;
        cv::Rect box;
    };

//    std::vector<YoloLayerData> layers{
//            {"446",    16, {{30,  61}, {62,  45},  {59,  119}}, 40},
//            {"output", 8,  {{10,  13}, {16,  30},  {33,  23}},  80},
//            {"448",    32, {{116, 90}, {156, 198}, {373, 326}}, 20},
//    };
    std::vector<YoloLayerData> layers{
            {"329",    16, {{30,  61}, {62,  45},  {59,  119}}, 40},
            {"output", 8,  {{10,  13}, {16,  30},  {33,  23}},  80},
            {"331",    32, {{116, 90}, {156, 198}, {373, 326}}, 20},
    };

    float im_scale;
};


#endif //ONETRY_CAMERAAPP_H
