#ifndef EDGESDETECT_H
#define EDGESDETECT_H

#include "config.h"
#include "model.h"
#include "chnsFunctions.h"

//void edgesDetect(_matrix<unsigned char>& img, Model& model, _matrix<float> E, _matrix<float> O);

void edgesDetect(Model& model, float* I, float* chns, float *chnsSs, int* sizeImg, _matrix<float>& E_m, _matrix<int>& ind_m);
void edgesDetect(_matrix<unsigned char>& I, Model& model, _matrix<float>& E, _matrix<float>& O);


void edgesChns(_matrix<unsigned char>& I, Model& model, _matrix<float>& chnsReg, _matrix<float>& chnsSim);


void rgbConvert(_matrix<unsigned char>& src,_matrix<float>& dst,int flag,int useSingle, const bool useSSE = false);

void convTri(_matrix<float>& src, _matrix<float>& dst, int grdSmooth);

void imPad(_matrix<unsigned char>& src, _matrix<unsigned char>& dst, int* p,int flag);

void edgesDetect(Model& model, _matrix<float>& I, _matrix<float>& chnsReg, _matrix<float>& chnsSim, _matrix<float>& E, _matrix<int>& inds);

void gradientMag(_matrix<float>& I, _matrix<float>& M, _matrix<float>& O, int channel, int normRad, float normConst);

void gradientHist(_matrix<float>& M, _matrix<float>& O, _matrix<float>& H, int binSize, int nOrients);

void multipleValue(_matrix<float>& mat, int startY, int endY, int startX, int endX, float t);

void calcO(_matrix<float>& E,_matrix<float>& O);


#endif

