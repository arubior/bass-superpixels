#include <iostream>
#include <string>
#include "test.hpp"
#include "../../include/model.h"
#include "../../include/edgesDetect.h"
#include <opencv2/opencv.hpp>
#include <istream>
#include <fstream>

#define PRINT(msg) std::cout<<__FILE__<<" "<<__LINE__<<" " <<msg<<std::endl

void mat2uchar_(_matrix<unsigned char>& img, cv::Mat t)
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


void loadImage(std::string imfilename, _matrix<unsigned char> &I){
    cv::Mat im = cv::imread(imfilename);
    mat2uchar_(I, im);
}

template <class numtype>
bool test_or_store(_matrix<numtype> &M, std::string filename){
    std::ifstream ios;
    ios.open(filename.c_str());
    if (ios.is_open()){ // compare
        _matrix<numtype> M2;
        boost::archive::binary_iarchive iar(ios);
        iar >> M2;
        return (M == M2);
    } else { // store
        std::ofstream ofs(filename.c_str());
        boost::archive::binary_oarchive oar(ofs);
        oar << M;
        ofs.close();
        test_or_store(M, filename);
    }
    ios.close();
}


void test_rgb2luv_setup(){

    float minu, minv, un, vn, mr[3], mg[3], mb[3];

    float nrm = 1.0f / 255;
    float *lut = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
    _matrix<float> _lut;
    _lut.dim = 2;
    _lut.size[0] = 1;
    _lut.size[1] = 1064;
    _lut.reShape(_lut.dim, _lut.size);
    memcpy(_lut.value, lut, sizeof(float) * _lut.num);

    PRINT(__FUNCTION__ +  std::string(" 'rgb2luv_setup'"));
    assert(test_or_store(_lut, "test_rgb2luv_setup.txt"));
}

void test_rgbConvert(){

    _matrix<unsigned char> I;
    loadImage("../../../python/girl-dress.jpeg", I);

    PRINT(__FUNCTION__ +  std::string(" 'normalize'"));
    _matrix<float> Inormalized(I.dim, I.size);
    rgbConvert(I, Inormalized, 1, 1);
    assert(test_or_store(Inormalized, "test_rgbConvertNormalized.txt"));

    PRINT(__FUNCTION__ +  std::string(" 'gray'"));
    _matrix<float> Igray(I.dim, I.size);
    rgbConvert(I, Igray, 0, 1);
    assert(test_or_store(Igray, "test_rgbConvertGray.txt"));


    PRINT(__FUNCTION__ +  std::string(" 'xyz'"));
    _matrix<float> Ixyz(I.dim, I.size);
    rgbConvert(I, Ixyz, 4, 1);
    assert(test_or_store(Ixyz, "test_rgbConvertXYZ.txt"));

    PRINT(__FUNCTION__ +  std::string(" ' xyz SSE'"));
    rgbConvert(I, Ixyz, 4, 1, true);
    assert(test_or_store(Ixyz, "test_rgbConvertXYZ.txt"));

    PRINT(__FUNCTION__ +  std::string(" 'luv'"));
    _matrix<float> luv(I.dim, I.size);
    rgbConvert(I, luv, 2, 1, false);
    assert(test_or_store(luv, "test_rgbConvertLuv.txt"));


    PRINT(__FUNCTION__ +  std::string(" 'luv SSE'"));
    rgbConvert(I, luv, 2, 1, true);
    assert(test_or_store(luv, "test_rgbConvertLuv.txt"));

    PRINT(__FUNCTION__ +  std::string(" 'hsv'"));
    _matrix<float> hsv(I.dim, I.size);
    rgbConvert(I, hsv, 3, 1);
    assert(test_or_store(hsv, "test_rgbConvertHsv.txt"));
}



void test_gradMag(){

    _matrix<unsigned char> Iu8;
    loadImage("../../../python/girl-dress.jpeg", Iu8);

    _matrix<float> I;
    I.reShape(Iu8.dim, Iu8.size);
    for (int i=0; i<Iu8.num; i++)
        I.value[i] = Iu8.value[i];

    _matrix<float> M, O;
    int size[3];
    size[0] = I.size[0];
    size[1] = I.size[1];
    size[2] = 1;

    M.reShape(I.dim, size);
    O.reShape(I.dim, size);
    gradMag(I.value, M.value, O.value, I.size[0], I.size[1], I.size[2],0);

    PRINT(__FUNCTION__ +  std::string(" 'M'"));
    assert(test_or_store(M, __FUNCTION__ + std::string("_M.txt")));

    PRINT(__FUNCTION__ +  std::string(" 'O'"));
    assert(test_or_store(O, __FUNCTION__ + std::string("_O.txt")));
}

void test_gradMagNorm(){

    _matrix<unsigned char> Iu8;
    loadImage("../../../python/girl-dress.jpeg", Iu8);

    _matrix<float> I;
    I.reShape(Iu8.dim, Iu8.size);
    for (int i=0; i<Iu8.num; i++)
        I.value[i] = Iu8.value[i];

    _matrix<float> M, O;
    int size[3];
    size[0] = I.size[0];
    size[1] = I.size[1];
    size[2] = 1;

    M.reShape(I.dim, size);
    O.reShape(I.dim, size);
    gradMag(I.value, M.value, O.value, I.size[0], I.size[1], I.size[2],0);

    _matrix<float> S;
    S.reShape(I.dim, I.size);
    convTri(M, S, 4);

    PRINT(__FUNCTION__ + std::string(" 'S'"));
    assert(test_or_store(S, __FUNCTION__ + std::string("_S.txt")));

    gradMagNorm_sse(M.value, S.value, M.size[0], M.size[1], 0.01f);

    PRINT(__FUNCTION__ + std::string(" 'M'"));
    assert(test_or_store(M, __FUNCTION__ + std::string("_M.txt")));
}


void test_gradientHist(){

    _matrix<unsigned char> Iu8;
    loadImage("../../../python/girl-dress.jpeg", Iu8);

    _matrix<float> I;
    I.reShape(Iu8.dim, Iu8.size);
    for (int i=0; i<Iu8.num; i++)
        I.value[i] = Iu8.value[i];

    _matrix<float> M, O;
    int size[3];
    size[0] = I.size[0];
    size[1] = I.size[1];
    size[2] = 1;

    M.reShape(I.dim, size);
    O.reShape(I.dim, size);
    gradMag(I.value, M.value, O.value, I.size[0], I.size[1], I.size[2],0);

    _matrix<float> S;
    S.reShape(I.dim, I.size);
    convTri(M, S, 4);

    gradMagNorm_sse(M.value, S.value, M.size[0], M.size[1], 0.01f);

    _matrix<float> H;
    int shrink = 2;
    int s = 1;
    int nOrients = 4;
    int binSize = 1 > ((float)shrink / (float)s) ? 1 : ((float)shrink / (float)s);
    gradientHist(M, O, H, binSize, nOrients);

    PRINT(__FUNCTION__ + std::string(" 'H'"));
    assert(test_or_store(H, __FUNCTION__ + std::string("_H.txt")));
}


void test_edgesChns(){
    Model model;
    model.initmodel("../../../models/bsd_model.bin");

    _matrix<unsigned char> Iu8;
    loadImage("../../../python/girl-dress.jpeg", Iu8);

    _matrix<unsigned char> It;
    int r = model.opts.imWidth / 2;
    int p[4] = {r,r,r,r};
    p[1] = p[1] + (4 - (Iu8.size[0] + 2 * r) % 4) % 4;
    p[3] = p[3] + (4 - (Iu8.size[1] + 2 * r) % 4) % 4;
    imPad(Iu8, It, p, 2);

    _matrix<float>chnsReg;
    _matrix<float>chnsSim;
    edgesChns(It, model, chnsReg, chnsSim);
    PRINT(__FUNCTION__ + std::string(" 'chnsReg'"));
    assert(test_or_store(chnsReg, __FUNCTION__ + std::string("_chnsReg.txt")));

    PRINT(__FUNCTION__ + std::string(" 'chnsSim'"));
    assert(test_or_store(chnsReg, __FUNCTION__ + std::string("_chnsSim.txt")));
}


void test_calcO(){

}


void test_edgesDetect(){
    _matrix<unsigned char> Iu8;
    loadImage("../../../python/girl-dress.jpeg", Iu8);


    Model model;
    model.initmodel("../../../models/bsd_model.bin");

    _matrix<float> E;
    _matrix<float> O;

    model.opts.multiscale = 0;
    model.opts.sharpen = 2;
    model.opts.nThreads = 4;

    edgesDetect(Iu8, model, E, O);

    PRINT(__FUNCTION__ + std::string(" 'edgesDetect_O'"));
    assert(test_or_store(O, __FUNCTION__ + std::string("_O.txt")));

    PRINT(__FUNCTION__ + std::string(" 'edgesDetect_E'"));
    assert(test_or_store(E, __FUNCTION__ + std::string("_E.txt")));
}

void test_convTri(){
    _matrix<unsigned char> img;
    loadImage("../../../python/girl-dress.jpeg", img);

    _matrix<float> I1;
    I1.reShape(img.dim, img.size);
    rgbConvert(img, I1, 1, 1);

    _matrix<float> r1;
    convTri(I1, r1, 1);
    PRINT(__FUNCTION__ + std::string(" 'r'"));
    assert(test_or_store(r1, __FUNCTION__ + std::string("_r.txt")));
}

void test_all(){
    test_rgb2luv_setup();
    test_rgbConvert();
    test_gradMag();
    test_gradMagNorm();
    test_gradientHist();
    test_convTri();
    test_edgesChns();
    test_edgesDetect();
}
