#include "CaffeCls.hpp"

SceneFeatureCls::SceneFeatureCls() {
    set_means(102.9801, 115.9465, 122.7717);
    set_feature_name("pool5/7x7_s1");
}


SceneFeatureCls::~SceneFeatureCls() {
    if (img_input_ != NULL)
        delete []img_input_;
}


void SceneFeatureCls::set_feature_name(string feature_name) {
    feature_blob_name_ = feature_name;
}


bool SceneFeatureCls::find_blobs() {
    int feature_layer_idx = -1;
    for(int i=0;i<(int)caffe_test_net_->layer_names().size();i++){
        if(caffe_test_net_->layer_names()[i] == feature_blob_name_)
            feature_layer_idx = i;
    }

    if (feature_layer_idx == -1) {
        cerr << "can not find feature layer named: " << feature_blob_name_ << endl;
        return false;
    }


    data_blob_ = caffe_test_net_->top_vecs()[0][0];
    feature_blob_ = caffe_test_net_->top_vecs()[feature_layer_idx][0];


    if (data_blob_->num() != 1) {
        cerr << "data batch size must be 1" << endl;
        return false;
    }
    if (data_blob_->channels() != 3) {
        cerr << "illegal data blob" << endl;
        return false;
    }

    data_layer = dynamic_cast<MemoryDataLayer<float>* >(caffe_test_net_->layers()[0].get());

    return true;
}


void SceneFeatureCls::get_input_data(const Mat img, int width, int height) {
    Mat patch = img;
    cv::resize( patch , patch, cv::Size(width, height) );

    patch.convertTo(patch, CV_32FC3);

    img_input_ = new float[width * height * 3];

    for(int i = 0 ; i < width*height ; i++) {
        img_input_[i] = ((float*)patch.data)[i*3] - pixel_means_[0];
        img_input_[width*height + i] = ((float*)patch.data)[i*3+1] - pixel_means_[1];
        img_input_[width*height*2 + i] = ((float*)patch.data)[i*3+2] - pixel_means_[2];
    }
}


vector<float> SceneFeatureCls::extract_features(const Mat imgin) {
    vector<float> fea_data;
    if (!find_blobs()) return fea_data;

    get_input_data(imgin, data_blob_->width(), data_blob_->height());

    vector<Blob<float>*> dummy_blob_input_vec;
    float dummy_label[1] = {0};
    data_layer->Reset(img_input_, dummy_label, 1);
    caffe_test_net_->Forward(dummy_blob_input_vec);


    for (int k=0; k<feature_blob_->count(); k++)
        fea_data.push_back(feature_blob_->cpu_data()[k]);

    return fea_data;
}



//-------------- interface ---------------
namespace person_craft {
    void* scene_feature_load_model(string proto_name, string model_name, string feature_name) {
        SceneFeatureCls* ptr = new SceneFeatureCls;
        ptr->set_model(proto_name, model_name);
        ptr->set_feature_name(feature_name);
        return (void*)ptr;
    }

    vector<float> extract_scene_features(const Mat img, void* ptr) {
        SceneFeatureCls* p = (SceneFeatureCls*)ptr;
        return p->extract_features(img);
    }
}


/*
using namespace person_craft;
int main() {
    Mat img = imread("20081122_2.jpg");
    if (img.empty()) {
        cout << "image not found!" << endl;
        return 0;
    }

    void* ptr = scene_feature_load_model("scene_extraction.prototxt", "GoogLeNet_scene_extraction.caffemodel");


    vector<float> fea = extract_scene_features(img, ptr);

    for(int i=0; i<fea.size(); i++)
        cout << fea[i] << endl;

    delete ptr;
}
*/

