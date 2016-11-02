#ifndef CAFFE_CLASS
#define CAFFE_CLASS

#include "caffe/caffe.hpp"
//#include "/home/wideeyes/caffe/include/caffe/caffe.hpp"
//#include "/home/arubio/caffe-fast-rcnn/include/caffe/caffe.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "structureEdge.h"
#include "person_craft.h"
using namespace caffe;
using namespace std;
using namespace cv;

struct Bbox {
    float rootcoord[4];
    float score;
};

struct PropBox {
    float x1, y1, x2, y2;
    float confidence;
    int cls_id;
};

//------------- Base Class ---------------
class CaffeCls {
public:
    CaffeCls()
    {
        caffe_test_net_ = NULL;
    }
    void set_model(string proto_name, string model_name, const bool use_gpu = false, const int gpu_id = 0){
        this->use_gpu_ = use_gpu;
        if (!use_gpu)
            Caffe::set_mode(Caffe::CPU);
        else{
            Caffe::SetDevice(gpu_id);
            Caffe::set_mode(Caffe::GPU);
        }
        //initial test net
        NetParameter test_net_param;
        ReadNetParamsFromTextFileOrDie(proto_name, &test_net_param);
        caffe_test_net_ = new Net<float>(test_net_param);
        printf("init test net proto ok\n");

        //read in pretrained net
        NetParameter trained_net_param;
        ReadNetParamsFromBinaryFileOrDie(model_name, &trained_net_param);
        caffe_test_net_->CopyTrainedLayersFrom(trained_net_param);
        printf("init test net prarameters ok\n");
    }

    void set_means(float b, float g, float r){
        pixel_means_[0] = b;
        pixel_means_[1] = g;
        pixel_means_[2] = r;
    }

    inline bool use_gpu(){
        return this->use_gpu_;
    }

    virtual ~CaffeCls() {
        if (caffe_test_net_ != NULL)
            delete caffe_test_net_;
    };

protected:
    Net<float>* caffe_test_net_;
    float pixel_means_[3];
    bool use_gpu_;
};


//-------------------- FastRCNN Class -----------------------
class FastRCNNCls: virtual public CaffeCls {
public:
    FastRCNNCls();
    ~FastRCNNCls();

    bool init_box_model(string);

    // shared memory method
    void get_boxes(Mat img);
    vector<PropBox> detect(const Mat, vector<int>);

    void set_overlap_th(float);
    void set_confidence_th(float);


    // non shared memory method
    std::vector<PropBox> detect(const Mat src, const std::vector<std::vector<float> > bboxes, const vector<int> cls);

    // return bbox candidates: [x y w h]
    void get_boxes(Mat img, vector<vector<float> > &boxes, const int nMaxBoxes = 3000);

    void plotresult(const Mat &imgin, PropBox &bbox);
    void set_boxes(int nBoxes, int nDim, float* pointer);

private:
    void mat2uchar(_matrix<unsigned char>& img, Mat t);

    // shared memory methods
    void image_pyramid(const Mat img);
    void project_im_rois();
    void do_forward();
    vector<PropBox> post_process(const vector<int> tgt_cls);


    // non shared memory methods
    void image_pyramid(const cv::Mat img, float *pixel_means,
                       float **img_output, std::vector<int> &img_shape,
                       float &scale_factor);

    void project_im_rois(std::vector<std::vector<float> > boxes,
                                           const float scale_factor,
                                           float **roi_input, std::vector<int> &roi_shape);

    void do_forward(const float *img_input, std::vector<int> img_shape,
                    const float *roi_input, std::vector<int> roi_shape,
                    std::vector<Blob<float>*> &result);

    std::vector<PropBox> post_process(const std::vector<Blob<float>*> detections,
                                      const std::vector<std::vector<float> > bbox_proposals,
                                      const vector<int> tgt_cls);


    bool nms(vector<Bbox> &src, vector<Bbox> &dst, float overlap);

    vector<vector<float> > boxes_;  // x1, y1, x2, y2
    vector<Blob<float>*> output_blobs;
    int box_num_;
    double scale_factor_;
    float* img_input_;
    float* roi_input_;
    vector<int> img_shape_;
    vector<int> roi_shape_;
    float overlap_th_;
    float confidence_th_;

    Model* box_model;
};



//----------------- Scene Feature Class ----------------
class SceneFeatureCls: virtual public CaffeCls {
public:
    SceneFeatureCls();
    ~SceneFeatureCls();
    void set_feature_name(string);
    vector<float> extract_features(const Mat);

private:
    bool find_blobs();
    void get_input_data(const Mat, int, int);

    Blob<float>* feature_blob_;
    Blob<float>* data_blob_;
    MemoryDataLayer<float>* data_layer;
    float* img_input_;
    string feature_blob_name_;
};


#endif
