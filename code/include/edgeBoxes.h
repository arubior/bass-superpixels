#ifndef EDGEBOXES_H
#define EDGEBOXES_H

#include "config.h"
#include "model.h"
#include <vector>

using namespace std;

class bb{
public:
	float coord[4];
	float score;
};

class _bbs{
public:
	vector<bb> bbs;
	int size;
};
void edgeBoxes(float*EE, float* OO, int h, int w, float _alpha,float _beta,float _minScore,float _maxBoxes,float _edgeMinMag,float _edgeMergeThr,float _clusterMinMag,
			   float _maxAspectRatio, float _minBoxArea, float _gamma, float _kappa,float *bbs);

void edgeBoxes(float*EE, float* OO, int h, int w, float _alpha, float _beta, float _minScore, float _maxBoxes, float _edgeMinMag, float _edgeMergeThr, float _clusterMinMag,
	float _maxAspectRatio, float _minBoxArea, float _gamma, float _kappa, vector<bb>& bbs);

#endif

