#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include "chnsFunctions.h"
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include <cmath>
#include <typeinfo>
#include <opencv2/opencv.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/exception/all.hpp>
#include <boost/exception/diagnostic_information.hpp>

using namespace std;

template <class numtype>
class _matrix{
public:
    numtype *value = NULL;
    int size[256];
    int dim;
    int num;
public:
    _matrix(){
        value = NULL;
        num = 0;
    }
    _matrix(int a){
        dim = a;
        num = 0;
        value = NULL;
    }
    _matrix(int a, int* b){
        dim = a;
        num = 1;
        for(int i=0;i<a;i++){
            size[i] = b[i];
            num *= size[i];
        }

        if (value== NULL){
            value = new numtype[num];
        }
        else{
            delete [] value;
            value = NULL;
            value = new numtype[num];
        }
    }
    _matrix(int a, int* b, numtype* data){
        _matrix(a, b);
        for (int i = 0;i<num;i++){
            value[i] = data[i];
        }
    }
    ~_matrix(){
        if (value!=NULL){
            delete [] value;
            value = NULL;
        }
    }
    void reShape(int a, int* b){
        dim = a;
        num = 1;
        for(int i=0;i<a;i++){
            size[i] = b[i];
            num *= size[i];
        }
        if (value== NULL){
            value = new numtype[num];
        }
        else{
            delete [] value;
            value = NULL;
            value = new numtype[num];
        }
    }
    void setValue(numtype t){
        for (int i = 0; i < num; i++){
            value[i] = t;
        }
    }
    void Init(int a, int* b, numtype* data){
        this->reShape(a, b);
        for (int i = 0;i<num;i++){
            value[i] = data[i];
        }
    }

    void Init(fstream& filein)
    {
        int a;
        filein.read((char*)& a,sizeof(int));
        int* b = new int[a];
        filein.read((char*)b,sizeof(int)*a);
        this->reShape(a, b);
        filein.read((char*)value,sizeof(numtype)*num);
        delete []b;
    }


    void Load(std::string filename){
        std::ifstream ff(filename.c_str());

        ff>>this->dim;
        for (int i=0; i<this->dim; i++)
            ff>>this->size[i];
        this->num = 1;
        for (int i=0; i<this->dim; i++)
            this->num *= this->size[i];
        this->value = new numtype[this->num];
        for (int i=0; i<this->num; i++){
            unsigned char aa;
            ff>>aa;
            (this->value)[i] = aa;
        }
        ff.close();
    }

    void Save(std::string filename){

        std::ofstream ff(filename.c_str());

        ff<<this->dim<<std::endl;
        for (int i=0; i<this->dim; i++){
            ff<<(int)this->size[i]<<" ";
        }
        ff<<std::endl;

        if (typeid(numtype) == typeid(unsigned char)){
            for (int i=0; i<this->num; i++)
                ff<<this->value[i]<<" ";
        } else if (typeid(this->value[0]) == typeid(float)){
            for (int i=0; i<this->num; i++)
                ff<<this->value[i]<<" ";
        }
        ff.close();
    }




    void Save2YML(std::string filename){

        cv::FileStorage ff(filename, cv::FileStorage::WRITE);
        int rows, cols = 1;
        for (int i=0; i<this->dim; i++){
            if (i==1){
                rows = this->size[i];
            }else{
                cols *= this->size[i];
            }
            char t[10];
            sprintf(t, "dim%d", i);
            ff<<t<<(int)this->size[i];
        }
        cv::Mat tmp(rows, cols, CV_32F, this->value);
        ff<<"data32f"<<tmp;
        cv::Mat tmp2(rows, cols, CV_8U, this->value);
        ff<<"data8u"<<tmp2;
        ff.release();
    }

    _matrix<numtype>& operator=(const _matrix<numtype>& t1){
        dim = t1.dim;
        num = t1.num;
        for (int i = 0; i < dim; i++){
            size[i] = t1.size[i];
        }

        if (value == NULL){
            value = new numtype[num];
        }
        else{
            delete[] value;
            value = NULL;
            value = new numtype[num];
        }

        for (int i = 0; i < num; i++){
            value[i] = t1.value[i];
        }
        return *this;
    }

    bool operator == (const _matrix<numtype>& t1){
        if (this->dim != t1.dim){
            std::cout<<"dim error: "<<this->dim<<" vs "<<t1.dim<<std::endl;
            return false;
        }

        if (this->num != t1.num){
            std::cout<<"num error: "<<this->num<<" vs "<<t1.num<<std::endl;
            return false;
        }

        for (int i = 0; i < dim; i++){
            if (this->size[i] != t1.size[i]){
                std::cout<<"size error: "<<this->size[i]<<" vs "<<t1.size[i]<<std::endl;
                return false;
            }
        }


        if ( (this->value == NULL) ^ (t1.value == NULL)){
            std::cout<<"pointer error: "<<std::endl;
            return false;
        }

        if (this->value != NULL && t1.value != NULL){

            int step_size = this->size[0] * this->size[1];

            int nChannels = this->dim == 3 ? this->size[2] : 1;

            float diff = 0.f;
            for (int t = 0; t<nChannels; t++){
                bool found_error = false;

                int i=0;
                while ( i < this->size[0] * this->size[1] ){
                    int idx = t * step_size + i;
                    float dist = (float)(this->value[idx]) - (float)(t1.value[idx]);
                    dist = fabs(dist / ((float)(this->value[idx]) + 1e-36f));
                    diff += dist;
                    if (dist > 1e-3){
                        std::cout<<(float)(this->value[idx])<<" vs "<< (float)(t1.value[idx])<<std::endl;
                        std::cout<<"channel: "<< t<<", index: "<< i <<", relative error:"<<dist<<endl;
                        std::cout.flush();
                        found_error = true;
                    }
                    i++;
                    /*
                    if (found_error)
                        break;
                        */
                }

                if (found_error){
                    std::cout<<"relative diff per pixel is: "<<diff/this->num<<std::endl;
                    cv::Mat d;
                    if (typeid(numtype) == typeid(float)){
                        cv::Mat a(this->size[1], this->size[0], CV_32FC1, this->value + t * this->size[0] * this->size[1]);
                        cv::Mat b(this->size[1], this->size[0], CV_32FC1, t1.value + t * this->size[0] * this->size[1]);
                        cv::absdiff(a, b, d);
                        cv::imshow("origin", a);
                        cv::imshow("b", b);
                        cv::imshow("d", d*100);
                    } else if (typeid(numtype) == typeid(unsigned char)){
                        cv::Mat a(this->size[1], this->size[0], CV_8UC1, this->value + t * this->size[0] * this->size[1]);
                        cv::Mat b(this->size[1], this->size[0], CV_8UC1, t1.value + t * this->size[0] * this->size[1]);
                        cv::absdiff(a, b, d);
                        cv::imshow("origin", a);
                        cv::imshow("b", b);
                        cv::imshow("d", d*100);
                    }
                    cv::waitKey(0);
                    return false;
                }
            }
            std::cout<<"relative diff per pixel is: "<<diff/this->num<<std::endl;
        }
        return true;
    }

    void show(std::string window_name){
        int t = 0; // channel 0
        cv::Mat a(this->size[1], this->size[0], CV_32FC1, this->value + t * this->size[0] * this->size[1]);
        cv::Mat b;
        cv::transpose(a,b);
        cv::imshow(window_name, b);
        cv::waitKey(0);
    }

    void cat3(_matrix<numtype>* t,int count)
    {
        dim = t[0].dim;
        size[0] = t[0].size[0];
        size[1] = t[0].size[1];
        size[2] = 0;
        for (int i = 0; i < count;i++){
            size[2] += t[i].size[2];
        }
        num = 1;
        for (int i = 0; i < dim; i++){
            num *= size[i];
        }
        if (value == NULL){
            value = new numtype[num];
        }
        else{
            delete[] value;
            value = NULL;
            value = new numtype[num];
        }
        int startIndex = 0;
        for (int i = 0; i < count; i++){
            for (int j = 0; j < t[i].num;j++){
                value[startIndex + j] = t[i].value[j];
            }
            startIndex += t[i].num;
        }
    }
};


std::ostream& operator <<(std::ofstream &os, const _matrix<unsigned char> data );
std::ostream& operator <<(std::ofstream &os, const _matrix<int> data );
std::ostream& operator <<(std::ofstream &os, const _matrix<float> data );

std::ifstream & operator >>(std::ifstream &is, const _matrix<unsigned char> data);
std::ifstream & operator >>(std::ifstream &is, const _matrix<int> data);
std::ifstream & operator >>(std::ifstream &is, const _matrix<float> data);

BOOST_SERIALIZATION_SPLIT_FREE(_matrix<unsigned char>)
namespace boost{
    namespace serialization{
    template<class Archive>
    void save(Archive &ar, const _matrix<unsigned char> &data, const unsigned int version){
        ar & data.dim;
        for (int i=0; i<data.dim; i++)
            ar & data.size[i];
        for (int i=0; i<data.num; i++)
            ar & (static_cast<unsigned char*>(data.value))[i];
    }

    template<class Archive>
    void load(Archive &ar, _matrix<unsigned char> &data, const unsigned int version){
        ar & data.dim;
        memset(data.size, 0, sizeof(int) * 256);
        for (int i=0; i<data.dim; i++)
            ar & data.size[i];
        data.reShape(data.dim, data.size);
        try{
            for (int i=0; i<data.num; i++)
                ar & (static_cast<unsigned char*>(data.value))[i];
        } catch (boost::exception &e){

            std::cout<<boost::diagnostic_information(e);
        }
    }
    }
}



BOOST_SERIALIZATION_SPLIT_FREE(_matrix<float>)
namespace boost{
    namespace serialization{
    template<class Archive>
    void save(Archive &ar, const _matrix<float> &data, const unsigned int version){
        ar & data.dim;
        for (int i=0; i<data.dim; i++)
            ar & data.size[i];
        for (int i=0; i<data.num; i++)
            ar & (static_cast<float*>(data.value))[i];
    }

    template<class Archive>
    void load(Archive &ar, _matrix<float> &data, const unsigned int version){
        ar & data.dim;
        memset(data.size, 0, sizeof(int) * 256);
        for (int i=0; i<data.dim; i++)
            ar & data.size[i];
        data.reShape(data.dim, data.size);
        try{
            for (int i=0; i<data.num * sizeof(float); i++)
                ar & ((unsigned char*)data.value)[i];
        } catch (boost::exception &e){

            std::cout<<boost::diagnostic_information(e);
        }
    }
    }
}


class _opts{
public:
    int imWidth;
    int gtWidth;
    int nPos;
    int nNeg;
    int nImgs;
    int nTrees;
    float fracFtrs;
    int minCount;
    int minChild;
    int maxDepth;
    int discretize;
    int nSamples;
    int nClasses;
    int split;
    int nOrients;
    int grdSmooth;
    int chnSmooth;
    int simSmooth;
    int normRad;
    int shrink;
    int nCells;
    int rgbd;
    int stride;
    int multiscale;
    int sharpen;
    int nTreesEval;
    int nThreads;
    int nms;
    int seed;
    int useParfor;
    int nChns;
    int nChnFtrs;
    int nSimFtrs;
    int nTotFtrs;


    float alpha;
    float beta;
    float minScore;
    int maxBoxes;
    float edgeMinMag;
    float edgeMergeThr;
    float clusterMinMag;
    float maxAspectRatio;
    float minBoxArea;
    float gamma;
    float kappa;


    void Init(fstream& filein){
        filein.read((char*)(&imWidth), sizeof(int));
        assert(imWidth == 32);
        filein.read((char*)&gtWidth,sizeof(int));
        assert(gtWidth == 16);
        filein.read((char*)&nPos,sizeof(int));
        assert(nPos == 500000);
        filein.read((char*)&nNeg,sizeof(int));
        assert(nNeg == 500000);
        filein.read((char*)&nImgs,sizeof(int));
        assert(nImgs == 2147483647);
        filein.read((char*)&nTrees,sizeof(int));
        assert(nTrees == 8);
        filein.read((char*)&fracFtrs,sizeof(float));
        assert(fracFtrs == 0.25f);
        filein.read((char*)&minCount,sizeof(int));
        assert(minCount == 1);
        filein.read((char*)&minChild,sizeof(int));
        assert(minChild == 8);
        filein.read((char*)&maxDepth,sizeof(int));
        assert(maxDepth == 64);
        //		filein.read((char*)&discretize,sizeof(int));
        //        assert(discretize == );
        filein.read((char*)&nSamples,sizeof(int));
        assert(nSamples == 256);
        filein.read((char*)&nClasses,sizeof(int));
        assert(nClasses == 2);
        //		filein.read((char*)&split,sizeof(int));
        //        assert(split == );
        filein.read((char*)&nOrients,sizeof(int));
        assert(nOrients == 4);
        filein.read((char*)&grdSmooth,sizeof(int));
        assert(grdSmooth == 0);
        filein.read((char*)&chnSmooth,sizeof(int));
        assert(chnSmooth == 2);
        filein.read((char*)&simSmooth,sizeof(int));
        assert(simSmooth == 8);
        filein.read((char*)&normRad,sizeof(int));
        assert(normRad == 4);
        filein.read((char*)&shrink,sizeof(int));
        assert(shrink == 2);
        filein.read((char*)&nCells,sizeof(int));
        assert(nCells == 5);
        filein.read((char*)&rgbd,sizeof(int));
        assert(rgbd == 0);

        filein.read((char*)&stride,sizeof(int));
        assert(stride == 2);
        filein.read((char*)&multiscale,sizeof(int));
        assert(multiscale == 0);
        filein.read((char*)&sharpen,sizeof(int));
        assert(sharpen == 2);
        filein.read((char*)&nTreesEval,sizeof(int));
        assert(nTreesEval == 4);

        filein.read((char*)&nThreads,sizeof(int));
        assert(nThreads == 4);
        filein.read((char*)&nms,sizeof(int));
        assert(nms == 0);
        filein.read((char*)&seed,sizeof(int));
        assert(seed == 1);
        filein.read((char*)&useParfor,sizeof(int));
        assert(useParfor == 0);
        filein.read((char*)&nChns,sizeof(int));
        assert(nChns == 13);
        filein.read((char*)&nChnFtrs,sizeof(int));
        assert(nChnFtrs == 3328);
        filein.read((char*)&nSimFtrs, sizeof(int));
        assert(nSimFtrs == 3900);
        filein.read((char*)&nTotFtrs,sizeof(int));
        assert(nTotFtrs == 7228);

        alpha = 0.65;
        beta = 0.75;
        minScore = 0.01;
        maxBoxes =  10000;
        edgeMinMag = 0.1;
        edgeMergeThr = 0.5;
        clusterMinMag = 0.5;
        maxAspectRatio = 3;
        minBoxArea = 1000;
        gamma = 2;
        kappa = 1.5;
    }
};

class Model {
public:
    _matrix<sType> thrs;
    _matrix<int> fids;
    _matrix<int> child;
    _matrix<int> count;
    _matrix<int> depth;
    _matrix<int> nSegs;
    _matrix<int> eBins;
    _matrix<int> eBnds;
    _matrix<unsigned char> segs;
    _opts opts;

    Model() {};
    Model(const string& n1)
    {
        initmodel(n1);
    };
    Model(fstream& n1)
    {
        initmodel(n1);
    };
    ~Model();
    void initmodel(const string&);
    void initmodel(fstream&);
};

#endif

