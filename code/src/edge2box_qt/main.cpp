#include <opencv2/opencv.hpp>
#include "CaffeCls.hpp"
#include <assert.h>
#include "test.hpp"

void mat2uchar(_matrix<unsigned char>& img, cv::Mat t)
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


bool mysort (PropBox a, PropBox b) {
    return (a.confidence > b.confidence);
}


bool is_equal(std::vector<std::vector<float> > v, std::vector<std::vector<float> > u){
    for (int i=0; i<v.size(); i++){
        std::vector<float> a = v[i];
        std::vector<float> b = u[i];
        for (int j=0; j<a.size(); j++){
            if (a[j] != b[j])
                return false;
        }
    }
    return true;
}


std::string bsd_model = "../../../models/bsd_model.bin";

/*
void test_edgesDetect(){

    cv::Mat im = cv::imread("../../../python/girl-dress.jpeg");

    _matrix<unsigned char> img;
    mat2uchar(img, im);

    Model* box_model = new Model();
    box_model->initmodel(bsd_model);

    _matrix<float> E1, E2;
    _matrix<float> O1, O2;

    edgesDetect(img, *box_model, E1, O1);
    edgesDetect(img, *box_model, E2, O2);

    std::cout<<"E is the same : "<< (E1 == E2)<<std::endl;
    std::cout<<"O is the same : "<< (O1 == O2)<<std::endl;
}
*/

void _edgesDetect(_matrix<unsigned char> &I, _matrix<float> &E1, _matrix<float> &O1){

    Model* box_model = new Model();
    box_model->initmodel(bsd_model);
    box_model->opts.sharpen = 2;
    box_model->opts.nThreads = 4;
    box_model->opts.multiscale = 0;
    edgesDetect(I, *box_model, E1, O1);
}


void test_gradientMag(){
    std::cout<<"Testing gradientMag()"<<std::endl;

    cv::Mat im = cv::imread("../../../python/girl-dress.jpeg");
    _matrix<unsigned char> img;
    mat2uchar(img, im);
    _matrix<float> I1;
    I1.reShape(img.dim, img.size);
    rgbConvert(img, I1, 2, 1);

    _matrix<float> I2;
    convTri(I1,I2,0);

    {
        std::ofstream ofs("test_gradientMag_I2.txt");
        boost::archive::binary_oarchive oar(ofs);
        oar << I2;
        ofs.close();

        for (int i=0; i<I2.dim; i++){
            cv::Mat a(I2.size[1], I2.size[0], CV_32FC1, I2.value + i * I2.size[0] * I2.size[1]);
            cv::imshow("I2", a);
            cv::waitKey(0);
        }
    }

    _matrix<float> M1, M2;
    _matrix<float> O1, O2;
    gradientMag(I2, M1, O1, 0, 4, 0.01f);

    gradientMag(I2, M2, O2, 0, 4, 0.01f);

    assert(M1 == M2);
    assert(O1 == O2);

    std::ofstream ofs("test_gradientMag_M.txt");
    {
    boost::archive::binary_oarchive oar(ofs);
    oar << M1;
    ofs.close();
    }


    ofs.open("test_gradientMag_O.txt");
    {
    boost::archive::binary_oarchive oar(ofs);
    oar << O1;
    ofs.close();
    }

}


void test_edgesDetect2(){

    cv::Mat im = cv::imread("../../../python/girl-dress.jpeg");

    _matrix<unsigned char> img;
    mat2uchar(img, im);

    Model* box_model = new Model();
    box_model->initmodel(bsd_model);

    int sizeE[3];
    sizeE[0] = img.size[0]; sizeE[1] = img.size[1]; sizeE[2] = 1;
    _matrix<float> E1, E2;
    E1.reShape(img.dim, sizeE);
    E2.reShape(img.dim, sizeE);
    E1.setValue(0);
    E2.setValue(0);


    int r = box_model->opts.imWidth / 2;
    int p[4] = {r,r,r,r};
    p[1] = p[1] + (4 - (img.size[0] + 2 * r) % 4) % 4;
    p[3] = p[3] + (4 - (img.size[1] + 2 * r) % 4) % 4;

    _matrix<unsigned char> It;
    imPad(img, It, p, 2);

    _matrix<float> chnsReg, chnsSim;
    {
        edgesChns(It, *box_model, chnsReg, chnsSim);
    }


    _matrix<float> I2;
    {
        _matrix<float> I1;
        I1.reShape(It.dim, It.size);
        rgbConvert(It, I1, 1, 1);
        convTri(I1, I2, 1);
    }

    _matrix<int> ind1, ind2;
    edgesDetect(*box_model, I2, chnsReg, chnsSim, E1, ind1);
    edgesDetect(*box_model, I2, chnsReg, chnsSim, E2, ind2);


    std::cout<<"edgeDetect2 is the same : "<< (E1 == E2)<<std::endl;
    std::cout<<"edgeDetect2 is the same : "<< (ind1 == ind2)<<std::endl;


    float t = (float)(box_model->opts.stride*box_model->opts.stride) / (float)box_model->opts.nTreesEval / (float)(box_model->opts.gtWidth*box_model->opts.gtWidth);
    r = box_model->opts.gtWidth / 2;
    int s = box_model->opts.sharpen;
    if (s == 0)
        t = t * 2;
    else if(s == 1)
        t = t*1.8;
    else
        t = t*1.66;

    multipleValue(E1, r, r - 1 + im.size[0], r, r - 1 + im.size[1], t);
    multipleValue(E2, r, r - 1 + im.size[0], r, r - 1 + im.size[1], t);
    std::cout<<"edgeDetect2 is the same : "<< (E1 == E2)<<std::endl;

    _matrix<float> E11, E22;
    convTri(E1, E11, 1);
    convTri(E2, E22, 1);
    E1 = E11;
    E2 = E22;
    std::cout<<"edgeDetect2 is the same : "<< (E11 == E22)<<std::endl;

    _matrix<float> O1, O2;
    calcO(E1, O1);
    calcO(E2, O2);


    std::cout<<"edgeDetect2 E1 the same : "<< (E1 == E2)<<std::endl;
    std::cout<<"edgeDetect2 O1 the same : "<< (O1 == O2)<<std::endl;
}


void test_edgesNMS(_matrix<unsigned char> &I, _matrix<float> &E){

    Model* box_model = new Model();
    box_model->initmodel(bsd_model);
    box_model->opts.sharpen = 2;
    box_model->opts.nThreads = 4;
    box_model->opts.multiscale = 1;

    _matrix<float> O;
    edgesDetect(I, *box_model, E, O);

    _matrix<float> E2;

    E2.reShape(E.dim,E.size);
    int len = 1;
    for (int i=0; i<E.dim; i++)
        len *= E.size[i];

    edgesNMS(E.value, O.value, 2, 0, 1, box_model->opts.nThreads, E.size[0], E.size[1], E2.value, len);
    E = E2;
}

void test_edgeBoxes(_matrix<unsigned char> &I, std::vector<bb> &bbs){

    Model* box_model = new Model();
    box_model->initmodel(bsd_model);
    box_model->opts.sharpen = 2;
    box_model->opts.nThreads = 4;
    box_model->opts.multiscale = 1;

    _matrix<float> E, O;
    edgesDetect(I, *box_model, E, O);

    _matrix<float> E2;

    E2.reShape(E.dim,E.size);
    int len = 1;
    for (int i=0; i<E.dim; i++)
        len *= E.size[i];

    edgesNMS(E.value, O.value, 2, 0, 1, box_model->opts.nThreads, E.size[0], E.size[1], E2.value, len);
    E = E2;

    edgeBoxes(E.value, O.value, E.size[0], E.size[1],
            box_model->opts.alpha, box_model->opts.beta,
            box_model->opts.minScore, box_model->opts.maxBoxes,
            box_model->opts.edgeMinMag, box_model->opts.edgeMergeThr,
            box_model->opts.clusterMinMag, box_model->opts.maxAspectRatio,
            box_model->opts.minBoxArea, box_model->opts.gamma, box_model->opts.kappa,
            bbs);
}

int main(int argc, char **argv){

    test_all();

    FastRCNNCls* fastRCNN = new FastRCNNCls; /* the proposal initializes */
    fastRCNN->init_box_model(argv[2]);
    bool use_gpu = true;
    fastRCNN->set_model(argv[3], argv[4], use_gpu, 0);

    cv::Mat im = cv::imread(argv[1]);

    _matrix<unsigned char> img;
    mat2uchar(img, im);

    std::vector<int> classes(54);
    for (int i=0; i<classes.size(); i++) {
        classes[i] = i+1;
    }
    std::vector<std::vector<float> > input_bboxes;

    std::ifstream g("boxes.txt");
    int cnt = 0;
    while (1){
        cnt++;
        if (cnt == 3000)
            break;
        float x1, y1, x2, y2;
        g>>x1>>y1>>x2>>y2;
        std::vector<float> bb;
        bb.push_back(x1);
        bb.push_back(y1);
        bb.push_back(x2);
        bb.push_back(y2);
        input_bboxes.push_back(bb);
    }
    g.close();

    bool first = true;
    std::vector<PropBox> detections;
    std::vector<std::vector<float> > bboxes;
    double total_time = 0;
    int iter = 0;
    for (; iter<100; iter++){


        double t1, t2;
        t1 = cvGetTickCount();
        std::vector<std::vector<float> > boxes;
        /* set the boxes for the image */
        fastRCNN->get_boxes(im, boxes);
        //boxes = input_bboxes;
        t2 = cvGetTickCount();

        //boxes.resize(13);
        std::cout<<"bboxes: "<<boxes.size()<<std::endl;
        std::cout.flush();

        /*
        std::cout<<"bboxes: "<<boxes.size()<<std::endl;
        for (int i=0; i<boxes.size(); i++){
            std::cout<<boxes[i][0]<<" "<<boxes[i][1]<<" "<<boxes[i][2]<<" "<<boxes[i][3]<<std::endl;
        }
*/
        std::cout<<(t2-t1)/(cvGetTickFrequency()*1000)<<" ms"<<std::endl;

        t1 = cvGetTickCount();
        std::vector<PropBox> pboxes = fastRCNN->detect(im, boxes, classes);
        t2 = cvGetTickCount();
        total_time += (t2-t1)/(cvGetTickFrequency()*1000.);
        std::sort(pboxes.begin(), pboxes.end(), mysort);

        for (int i=0; i<pboxes.size(); i++){
            PropBox p = pboxes[i];
            // std::cout<<p.cls_id<<" "<<p.x1<<" "<<p.y1<<" "<<p.x2<<" "<<p.y2<<" "<<p.confidence<<std::endl;
            // std::cout.flush();
        }

        if (first){
            detections.insert(detections.end(), pboxes.begin(), pboxes.end());
            bboxes.insert(bboxes.begin(), boxes.begin(), boxes.end());
            first = false;

            /*
            std::ofstream f("boxes.txt");
            for (int i=0; i<bb.size(); i++){
                std::vector<float> b1 = bb[i];
                for (int j=0; j<b1.size(); j++)
                    f<<b1[j]<<" ";
                f<<std::endl;
            }
            f.close();
            exit(0);
            */
        }else{

            assert(bboxes.size() == boxes.size());

            for (int i=0; i<bboxes.size(); i++){
                std::vector<float> a1 = boxes[i];
                std::vector<float> b1 = bboxes[i];
                for (int j=0; j<a1.size(); j++){
                    try{
                        if (a1[j] != b1[j])
                            throw std::logic_error( "testing logic_error" );
                    }
                    catch ( const std::logic_error & e ) {
                        std::cout<<a1[j]<<" "<<b1[j]<<std::endl;
                        exit(0);
                    }
                }
            }


            assert(detections.size() == pboxes.size());
            for (int i=0; i<detections.size(); i++){
                PropBox a1 = detections[i];
                PropBox b1 = pboxes[i];

                if (memcmp(&a1, &b1, sizeof(PropBox))!=0){
                    try{
                        throw std::logic_error( "testing logic_error" );
                    } catch (const std::logic_error & e ) {

                        std::cout<<a1.cls_id<<" "<<a1.x1<<" "<<a1.y1<<" "<<a1.x2<<" "<<a1.y2<<" "<<a1.confidence<<std::endl;
                        std::cout<<b1.cls_id<<" "<<b1.x1<<" "<<b1.y1<<" "<<b1.x2<<" "<<b1.y2<<" "<<b1.confidence<<std::endl;
                        std::cout.flush();
                        exit(0);
                    }
                }
            }
        }

        //        for (int i=0; i<pboxes.size(); i++) {
        //            printf("%f in the %d class\n", pboxes[i].confidence, pboxes[i].cls_id);
        //            fastRCNN->plotresult(im, pboxes[i]);
        //        }
    }
    std::cout<<"cnn elapsed time: "<<total_time /iter<<std::endl;
}
