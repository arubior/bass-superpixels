#include "CaffeCls.hpp"
#include <cmath>
#include <string>
#include <time.h>

struct _valindex {
    float value;
    int index;
    float area;
    float x1;
    float x2;
    float y1;
    float y2;
};


FastRCNNCls::FastRCNNCls() {
    set_means(102.9801, 115.9465, 122.7717);
    set_overlap_th(0.3);
    set_confidence_th(0.2);
    roi_input_ = NULL;
    img_input_ = NULL;
    box_model = NULL;
}


FastRCNNCls::~FastRCNNCls() {
    if (roi_input_ != NULL)
        delete []roi_input_;
    if (img_input_ != NULL)
        delete []img_input_;
    if(box_model != NULL)
    {
        delete box_model;
        box_model = NULL;
    }
}


void FastRCNNCls::mat2uchar(_matrix<unsigned char>& img, Mat t)
{
    int size[3];
    size[0] = t.rows;size[1] = t.cols;size[2] = 3;
    img.reShape(3,size);

    int* th = new int[size[1]];
    int* tw = new int[size[0]];

    for (int i = 0; i < size[1]; i++) th[i] = i*size[0];
    for (int i = 0; i < size[0]; i++) tw[i] = i*size[1]*3;

    for (int k = 0; k < size[2];k++){
        int ind1 = size[0] * size[1] * k;
        for (int j = 0; j < size[1];j++){
            int j3 = j * 3;
            for (int i = 0; i < size[0];i++){
                img.value[ind1 + th[j] + i] = t.data[tw[i] + j3 + 2 - k];
            }
        }
    }
    delete []th;
    delete []tw;
}

bool FastRCNNCls::init_box_model(string model_name)
{
    if(box_model != NULL) delete box_model;
    box_model = new Model();
    box_model->initmodel(model_name);
    return true;
}


void FastRCNNCls::set_boxes(int nBoxes, int nDim, float* pointer) {
    /* copy the boxes into it */
    if (nDim != 4) cerr << "Error reading the dimension!" <<" Its "<<nDim<<endl;

    box_num_ = nBoxes;

    boxes_.clear();
    boxes_.resize(box_num_);

    for (int i = 0; i < box_num_; i++) {
        boxes_[i].push_back(pointer[0]);
        boxes_[i].push_back(pointer[1]);
        boxes_[i].push_back(pointer[2]);
        boxes_[i].push_back(pointer[3]);
        pointer = pointer + 4;
    }
}
void FastRCNNCls::get_boxes(Mat img_mat) {
    vector<vector<float> > boxes_out;
    get_boxes(img_mat, boxes_out);
}

void FastRCNNCls::get_boxes(Mat img_mat, vector<vector<float> > &boxes_out, const int nMaxBoxes) {

    cv::Mat im_standard;

    int SS = 640;
    int WW = std::max(img_mat.cols, img_mat.rows);
    float scale = (float)SS / (float)WW;
    if (WW < 1200){
        im_standard = img_mat;
        scale = 1.f;
    }
    else
      cv::resize(img_mat, im_standard, cv::Size(), scale, scale);

    //transform img format
    _matrix<unsigned char> img;
    mat2uchar(img, im_standard);

    //get boxes_
    vector<bb> bbs; // x1, y1, width, height

    structureEdge(img, *box_model, bbs);

    /////////// ORIGINAL CODE BEGIN ////////////////////////
//    if (bbs.size() > 3000)
//        box_num_ = 3000;
//    else
//        box_num_ = bbs.size();
//    for (int i=0; i<bbs.size(); i++){
//        bbs[i].coord[0] /= scale;
//        bbs[i].coord[1] /= scale;
//        bbs[i].coord[2] /= scale;
//        bbs[i].coord[3] /= scale;
//    }

//    boxes_.clear();
//    boxes_.resize(box_num_);
//    for (int i=0; i<box_num_; i++) {
//        boxes_[i].push_back(bbs[i].coord[0]);
//        boxes_[i].push_back(bbs[i].coord[1]);
//        boxes_[i].push_back(bbs[i].coord[2] + bbs[i].coord[0] - 1);
//        boxes_[i].push_back(bbs[i].coord[3] + bbs[i].coord[1] - 1);
//    }

//    boxes_out.clear();
//    boxes_out.resize(box_num_);
//    for(int i=0;i < box_num_; i++)
//    {
//        boxes_out[i].assign(boxes_[i].begin(), boxes_[i].end());
//    }
    /////////// ORIGINAL CODE END ////////////////////////

    int nBoxes = std::min( (int)bbs.size(), nMaxBoxes);
    boxes_out.clear();
    boxes_out.resize(nBoxes);
    float iScale = 1.f / (scale + 1e-6f);
    for (int i=0; i<nBoxes; i++){
        boxes_out[i].push_back(bbs[i].coord[0] * iScale);
        boxes_out[i].push_back(bbs[i].coord[1] * iScale);
        boxes_out[i].push_back( (bbs[i].coord[2] + bbs[i].coord[0])* iScale - 1);
        boxes_out[i].push_back( (bbs[i].coord[3] + bbs[i].coord[1])* iScale - 1);
    }
}


void FastRCNNCls::image_pyramid(const Mat imgin) {
    // img must be a BGR image
    const float SCALES = 600;
    const float MAX_SIZE = 1000;

    Mat imgout;

    imgin.convertTo(imgout, CV_32FC3);
    //imshow("debug",imgout);

    for(int i = 0; i < imgout.cols*imgout.rows; i++) {
        ((float*)imgout.data)[i*3] -= pixel_means_[0];
        ((float*)imgout.data)[i*3+1] -= pixel_means_[1];
        ((float*)imgout.data)[i*3+2] -= pixel_means_[2];
    }

    float short_len = min(imgout.cols, imgout.rows);
    float long_len = max(imgout.cols, imgout.rows);

    scale_factor_ = SCALES / short_len;
    if (long_len > MAX_SIZE)
        scale_factor_ = MAX_SIZE / long_len;

    cv::resize(imgout, imgout, Size(), scale_factor_, scale_factor_);

    // copy image data to img_input_
    // the format must be transformed to channels * height * width
    if(img_input_ != NULL) {delete[] img_input_; img_input_ = NULL;}
    img_input_ = new float[imgout.cols * imgout.rows * imgout.channels()];
    for (int i=0; i<imgout.cols*imgout.rows; i++) {
        img_input_[i] = ((float*)imgout.data)[i*3];
        img_input_[imgout.cols*imgout.rows + i] = ((float*)imgout.data)[i*3+1];
        img_input_[imgout.cols*imgout.rows*2 + i] = ((float*)imgout.data)[i*3+2];
    }

    // get new img shape
    img_shape_.resize(4);
    img_shape_[0] = 1;
    img_shape_[1] = imgout.channels();
    img_shape_[2] = imgout.rows;
    img_shape_[3] = imgout.cols;
}

void FastRCNNCls::image_pyramid(const cv::Mat imgin,
                                float *pixel_means,
                                float **img_output, std::vector<int> &img_shape,
                                float &scale_factor){
    // img must be a BGR image
    const float SCALES = 600;
    const float MAX_SIZE = 1000;

    Mat imgout;

    imgin.convertTo(imgout, CV_32FC3);
    //imshow("debug",imgout);

    for(int i = 0; i < imgout.cols*imgout.rows; i++) {
        ((float*)imgout.data)[i*3] -= pixel_means[0];
        ((float*)imgout.data)[i*3+1] -= pixel_means[1];
        ((float*)imgout.data)[i*3+2] -= pixel_means[2];
    }

    float short_len = min(imgout.cols, imgout.rows);
    float long_len = max(imgout.cols, imgout.rows);

    scale_factor = SCALES / short_len;
    if (long_len > MAX_SIZE)
        scale_factor = MAX_SIZE / long_len;

    cv::resize(imgout, imgout, Size(), scale_factor, scale_factor);

    // copy image data to img_input_
    // the format must be transformed to channels * height * width
    *img_output = (float*)malloc(sizeof(float) * imgout.cols * imgout.rows * imgout.channels());
    for (int i=0; i<imgout.cols*imgout.rows; i++) {
        (*img_output)[i] = ((float*)imgout.data)[i*3];
        (*img_output)[imgout.cols*imgout.rows + i] = ((float*)imgout.data)[i*3+1];
        (*img_output)[imgout.cols*imgout.rows*2 + i] = ((float*)imgout.data)[i*3+2];
    }

    // get new img shape
    img_shape.resize(4);
    img_shape[0] = 1;
    img_shape[1] = imgout.channels();
    img_shape[2] = imgout.rows;
    img_shape[3] = imgout.cols;
}


void FastRCNNCls::project_im_rois() {
    int roi_data_len_ = box_num_ * 5;
    if(roi_input_ != NULL){ delete[] roi_input_; roi_input_ = NULL;}
    roi_input_ = new float[roi_data_len_];
    for (int i=0; i<box_num_; i++) {
        roi_input_[i*5] = 0;
        for (int j=1; j<=4; j++)
            roi_input_[i*5+j] = boxes_[i][j-1] * scale_factor_;
    }

    //get new roi shape
    roi_shape_.resize(4);
    roi_shape_[0] = box_num_;
    roi_shape_[1] = 5;
    roi_shape_[2] = 1;
    roi_shape_[3] = 1;
}



void FastRCNNCls::project_im_rois(std::vector<std::vector<float> > boxes,
                                  const float scale_factor,
                                  float **roi_input, std::vector<int> &roi_shape){

    int roi_data_len_ = boxes.size() * 5;

    *roi_input = (float*)malloc(sizeof(float) * roi_data_len_);
    for (int i=0; i<boxes.size(); i++) {
        (*roi_input)[i*5] = 0;
        for (int j=1; j<=4; j++)
            (*roi_input)[i*5+j] = boxes[i][j-1] * scale_factor;
    }

    //get new roi shape
    roi_shape.resize(4);
    roi_shape[0] = boxes.size();
    roi_shape[1] = 5;
    roi_shape[2] = 1;
    roi_shape[3] = 1;
}


void FastRCNNCls::set_overlap_th(float th) {
    overlap_th_ = th;
}

void FastRCNNCls::set_confidence_th(float th) {
    confidence_th_ = th;
}

bool rindcom(const _valindex& t1, const _valindex& t2) {
    return t1.value < t2.value;
}

bool FastRCNNCls::nms(vector<Bbox> &src, vector<Bbox> &dst, float overlap) {
    // pick out the most probable boxes
    if (src.empty())
        return false;
    int srcnum = src.size();
    vector<_valindex> s(srcnum);
    for (int i=0;i<srcnum;i++)
    {
        s[i].x1=src[i].rootcoord[0];
        s[i].y1=src[i].rootcoord[1];
        s[i].x2=src[i].rootcoord[2];
        s[i].y2=src[i].rootcoord[3];
        s[i].value=src[i].score;
        s[i].index=i;
        s[i].area=(src[i].rootcoord[2]-src[i].rootcoord[0]+1)*(src[i].rootcoord[3]-src[i].rootcoord[1]+1);
    }
    std::sort(s.begin(),s.end(),rindcom);
    vector<int>pick;

    while (!s.empty())
    {
        vector<int>suppress;
        int last=s.size()-1;
        pick.push_back(s[last].index);
        suppress.push_back(last);
        for (int pos=0;pos<last;pos++)
        {
            float xx1=max(s[last].x1,s[pos].x1);
            float yy1=max(s[last].y1,s[pos].y1);
            float xx2=min(s[last].x2,s[pos].x2);
            float yy2=min(s[last].y2,s[pos].y2);
            float w=xx2-xx1+1;
            float h=yy2-yy1+1;
            if (w>0&&h>0)
            {
                float o=w*h/s[pos].area;
                if (o>overlap)
                {
                    suppress.push_back(pos);
                }
            }
        }
        sort(suppress.begin(),suppress.end());
        for (int i=suppress.size()-1;i>=0;i--)
        {
            s.erase(s.begin()+suppress[i]);
        }
    }

    if (!dst.empty())
        dst.clear();

    for (int i=0;i<(int)pick.size();i++) {
        dst.push_back(src[pick[i]]);
    }

    return true;
}


void FastRCNNCls::do_forward() {
    const vector<Blob<float>*>& input_blobs = caffe_test_net_->input_blobs();

    if (input_blobs.size() != 2) {
        cerr << "illegal input layer!" << endl;
        return;
    }
    if (input_blobs[0]->num() != 1) {
        cerr << "batch size must be 1" << endl;
        return;
    }

    // import the image data and roi data
    input_blobs[0]->Reshape(img_shape_);
    input_blobs[1]->Reshape(roi_shape_);
    memcpy(input_blobs[0]->mutable_cpu_data(), img_input_, sizeof(float) * input_blobs[0]->count());
    memcpy(input_blobs[1]->mutable_cpu_data(), roi_input_, sizeof(float) * input_blobs[1]->count());

    // caffe net forward and get the output data
    output_blobs = caffe_test_net_->ForwardPrefilled();
}


void FastRCNNCls::do_forward(const float *img_input, std::vector<int> img_shape,
                             const float *roi_input, std::vector<int> roi_shape,
                             vector<Blob<float>*> &result){

    const std::vector<Blob<float> *> &input_blobs = caffe_test_net_->input_blobs();
    if (input_blobs.size() != 2) {
        cerr << "illegal input layer!" << endl;
        return;
    }
    if (input_blobs[0]->num() != 1) {
        cerr << "batch size must be 1" << endl;
        return;
    }
    input_blobs[0]->Reshape(img_shape);
    input_blobs[1]->Reshape(roi_shape);

    if (this->use_gpu())
        Caffe::set_mode(Caffe::GPU);
    else
        Caffe::set_mode(Caffe::CPU);

    switch (Caffe::mode()){
    case Caffe::CPU:
        caffe::caffe_copy(
                    input_blobs[0]->count(),
                    img_input,
                    input_blobs[0]->mutable_cpu_data()
                );
        caffe::caffe_copy(
                    input_blobs[1]->count(),
                    roi_input,
                    input_blobs[1]->mutable_cpu_data()
                );
        break;
    case Caffe::GPU:
        caffe::caffe_copy(
                    input_blobs[0]->count(),
                    img_input,
                    input_blobs[0]->mutable_gpu_data()
                );
        caffe::caffe_copy(
                    input_blobs[1]->count(),
                    roi_input,
                    input_blobs[1]->mutable_gpu_data()
                );
        break;
    }

    result = caffe_test_net_->ForwardPrefilled();
}


vector<PropBox> FastRCNNCls::post_process(const vector<int> tgt_cls) {

    vector<PropBox> prop_boxes;

    if (output_blobs.size() == 1) {
        int box_num = output_blobs[0]->num();
        int cls_num = output_blobs[0]->channels();

        for (int c=0; c<tgt_cls.size(); c++) {
            int j = tgt_cls[c];
            for (int i=0; i<box_num; i++) {
                if (output_blobs[0]->cpu_data()[i*cls_num+j] > confidence_th_) {
                    PropBox temp;
                    temp.cls_id = j;
                    temp.confidence = output_blobs[0]->cpu_data()[i*cls_num+j];
                    temp.x1 = boxes_[i][0];
                    temp.y1 = boxes_[i][1];
                    temp.x2 = boxes_[i][2];
                    temp.y2 = boxes_[i][3];
                    prop_boxes.push_back(temp);
                }
            }
        }
    }
    else if (output_blobs.size() == 2) {
        int box_num = output_blobs[1]->num();
        int cls_num = output_blobs[1]->channels();

        vector<vector<float> > prob;
        prob.resize(box_num);

        for (int i=0; i<box_num; i++) {
            prob[i].resize(cls_num);
            for (int j=0; j<cls_num; j++) {
                prob[i][j] = output_blobs[1]->cpu_data()[i*cls_num+j];
            }
        }

        const float* bbox_deltas = output_blobs[0]->cpu_data();

        vector<Bbox> bboxes(box_num);
        vector<Bbox> outboxes;

        // get the boxes after shifting and its confidence
        for (int c=0; c<tgt_cls.size(); c++) {
            int j = tgt_cls[c];
            if (j == this->output_blobs[0]->shape(1)>>2)
                continue;
            for	(int i=1; i<box_num; i++) {
                bboxes[i].score = prob[i][j];
                float src_w = boxes_[i][2] - boxes_[i][0];
                float src_h = boxes_[i][3] - boxes_[i][1];
                float src_ctr_x = boxes_[i][0] + 0.5 * src_w;
                float src_ctr_y = boxes_[i][1] + 0.5 * src_h;
                float dst_ctr_x = bbox_deltas[i*4*cls_num + j*4];
                float dst_ctr_y = bbox_deltas[i*4*cls_num + j*4+1];
                float dst_scl_x = bbox_deltas[i*4*cls_num + j*4+2];
                float dst_scl_y = bbox_deltas[i*4*cls_num + j*4+3];
                float pred_ctr_x = dst_ctr_x * src_w + src_ctr_x;
                float pred_ctr_y = dst_ctr_y * src_h + src_ctr_y;
                float pred_w = exp(dst_scl_x) * src_w;
                float pred_h = exp(dst_scl_y) * src_h;
                bboxes[i].rootcoord[0] = pred_ctr_x - 0.5 * pred_w;
                bboxes[i].rootcoord[1] = pred_ctr_y - 0.5 * pred_h;
                bboxes[i].rootcoord[2] = pred_ctr_x + 0.5 * pred_w;
                bboxes[i].rootcoord[3] = pred_ctr_y + 0.5 * pred_h;
            }

            if (!nms(bboxes, outboxes, overlap_th_)) {
                cerr << "bounding box is empty!" << endl;
            }

            for (int i=0; i<outboxes.size(); i++) {
                if (outboxes[i].score > confidence_th_) {
                    PropBox tmp;
                    tmp.x1 = outboxes[i].rootcoord[0];
                    tmp.y1 = outboxes[i].rootcoord[1];
                    tmp.x2 = outboxes[i].rootcoord[2];
                    tmp.y2 = outboxes[i].rootcoord[3];
                    tmp.confidence = outboxes[i].score;
                    tmp.cls_id = j;
                    prop_boxes.push_back(tmp);
                }
            }
        }
    }

    return prop_boxes;
}



std::vector<PropBox> FastRCNNCls::post_process(const std::vector<Blob<float>*> detections,
                                               const std::vector<std::vector<float> > bbox_proposals,
                                               const vector<int> tgt_cls){
    std::vector<PropBox> prop_boxes;
    if (detections.size() == 1){
        int box_num = detections[0]->num();
        int cls_num = detections[0]->channels();

        for (int c=0; c<tgt_cls.size(); c++) {
            int j = tgt_cls[c];
            for (int i=0; i<box_num; i++) {
                if (detections[0]->cpu_data()[i*cls_num+j] > confidence_th_) {
                    PropBox temp;
                    temp.cls_id = j;
                    temp.confidence = detections[0]->cpu_data()[i*cls_num+j];
                    temp.x1 = bbox_proposals[i][0];
                    temp.y1 = bbox_proposals[i][1];
                    temp.x2 = bbox_proposals[i][2];
                    temp.y2 = bbox_proposals[i][3];
                    prop_boxes.push_back(temp);
                }
            }
        }
    }else if (detections.size() == 2){
        int box_num = detections[1]->num();
        int cls_num = detections[1]->channels();
        vector<vector<float> > prob;
        prob.resize(box_num);

        for (int i=0; i<box_num; i++) {
            prob[i].resize(cls_num);
            for (int j=0; j<cls_num; j++) {
                prob[i][j] = detections[1]->cpu_data()[i*cls_num+j];
            }
        }

        const float* bbox_deltas = detections[0]->cpu_data();

        vector<Bbox> bboxes(box_num);
        vector<Bbox> outboxes;

        // get the boxes after shifting and its confidence
        for (int c=0; c<tgt_cls.size(); c++) {
            int j = tgt_cls[c];
            if (j == detections[0]->shape(1)>>2)
                continue;
            for	(int i=1; i<box_num; i++) {
                bboxes[i].score = prob[i][j];
                float src_w = bbox_proposals[i][2] - bbox_proposals[i][0];
                float src_h = bbox_proposals[i][3] - bbox_proposals[i][1];
                float src_ctr_x = bbox_proposals[i][0] + 0.5 * src_w;
                float src_ctr_y = bbox_proposals[i][1] + 0.5 * src_h;
                float dst_ctr_x = bbox_deltas[i*4*cls_num + j*4];
                float dst_ctr_y = bbox_deltas[i*4*cls_num + j*4+1];
                float dst_scl_x = bbox_deltas[i*4*cls_num + j*4+2];
                float dst_scl_y = bbox_deltas[i*4*cls_num + j*4+3];
                float pred_ctr_x = dst_ctr_x * src_w + src_ctr_x;
                float pred_ctr_y = dst_ctr_y * src_h + src_ctr_y;
                float pred_w = exp(dst_scl_x) * src_w;
                float pred_h = exp(dst_scl_y) * src_h;
                bboxes[i].rootcoord[0] = pred_ctr_x - 0.5 * pred_w;
                bboxes[i].rootcoord[1] = pred_ctr_y - 0.5 * pred_h;
                bboxes[i].rootcoord[2] = pred_ctr_x + 0.5 * pred_w;
                bboxes[i].rootcoord[3] = pred_ctr_y + 0.5 * pred_h;
            }

            if (!nms(bboxes, outboxes, overlap_th_)) {
                cerr << "bounding box is empty!" << endl;
            }

            for (int i=0; i<outboxes.size(); i++) {
                if (outboxes[i].score > confidence_th_) {
                    PropBox tmp;
                    tmp.x1 = outboxes[i].rootcoord[0];
                    tmp.y1 = outboxes[i].rootcoord[1];
                    tmp.x2 = outboxes[i].rootcoord[2];
                    tmp.y2 = outboxes[i].rootcoord[3];
                    tmp.confidence = outboxes[i].score;
                    tmp.cls_id = j;
                    prop_boxes.push_back(tmp);
                }
            }
        }
    }
    return prop_boxes;
}


vector<PropBox> FastRCNNCls::detect(const Mat img, const vector<int> tgt_cls) {

    image_pyramid(img);

//    float *im_input;
//    std::vector<int> im_shape;
//    image_pyramid(img, this->pixel_means_, &im_input, im_shape);
//    project_im_rois(this->)
    project_im_rois();
    do_forward();

    delete []this->img_input_;
    this->img_input_ = NULL;
    delete []this->roi_input_;
    this->roi_input_ = NULL;

    return post_process(tgt_cls);
}

std::vector<PropBox> FastRCNNCls::detect(const Mat src, const std::vector<std::vector<float> > bboxes, const vector<int> cls){

    if (bboxes.size() == 0){
        std::vector<PropBox> nothing;
        return nothing;
    }

    // image pyramid
    float *im_input;
    std::vector<int> im_shape;
    float scale_factor;
    image_pyramid(src, this->pixel_means_, &im_input, im_shape, scale_factor);

    // roi projection
    float *roi_input;
    std::vector<int> roi_shape;
    project_im_rois(bboxes, scale_factor,  &roi_input, roi_shape);

    std::vector<Blob<float>*> result;
    do_forward(im_input, im_shape, roi_input, roi_shape, result);

    free(im_input);
    free(roi_input);

    return post_process(result, bboxes, cls);
}


void FastRCNNCls::plotresult(const Mat &imgin, PropBox &bbox)
{
    int fontFace = FONT_HERSHEY_SIMPLEX;
    float fontScale = 0.5;
    int thickness = 1;

    Mat img = imgin;

    string ttext;
    char s[50]={0};
    sprintf(s,"P: %4f, cls: %d",bbox.confidence, bbox.cls_id);
    ttext=s;
    putText(img, ttext, Point((int)bbox.x1,(int)bbox.y1), fontFace, fontScale, Scalar::all(0), thickness,8);
    // rectangle(img, Point((int)bbox.x1,(int)bbox.y1), Point((int)bbox.x2,(int)bbox.y2),cvScalar(0,0,255),2,8, 0);


    imshow("debug", img);
    waitKey();
}
