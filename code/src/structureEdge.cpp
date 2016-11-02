// structureEdge.cpp : Defines the entry point for the console application.
//
#include <opencv2/opencv.hpp>
#include "structureEdge.h"

void structureEdge(_matrix<unsigned char>& img, Model& model, vector<bb>& bbs)
{
    _matrix<float> E;
    _matrix<float> O;

    model.opts.multiscale = MULTISCALE;
    model.opts.sharpen = 2;
    model.opts.nThreads = 4;

    edgesDetect(img, model, E, O);

    _matrix<float> E2;
    E2.reShape(E.dim,E.size);
    int len = 1;
    for (int i=0; i<E.dim; i++)
        len *= E.size[i];

    edgesNMS(E.value, O.value, 2, 0, 1, model.opts.nThreads, E.size[0], E.size[1], E2.value, len);
    E = E2;

   // edgeBoxes(E.value, O.value, E.size[0], E.size[1], model.opts.alpha, model.opts.beta, model.opts.minScore, model.opts.maxBoxes, model.opts.edgeMinMag, model.opts.edgeMergeThr, model.opts.clusterMinMag,
     //   model.opts.maxAspectRatio, model.opts.minBoxArea, model.opts.gamma, model.opts.kappa,bbs);
}

void structureEdge2(_matrix<unsigned char>& img, Model& model, _matrix<float> &E){

    _matrix<float> O;
    _matrix<float> Eb;
    // model.opts.multiscale = MULTISCALE;
    model.opts.sharpen = 2;
    model.opts.nThreads = 4;

    edgesDetect(img, model, Eb, O);
    E=Eb;

    // _matrix<float> E2;
    // E2.reShape(E.dim,E.size);
    // int len = 1;
    // for (int i=0; i<E.dim; i++)
    //     len *= E.size[i];

    // edgesNMS(E.value, O.value, 1, 5, 1.01, model.opts.nThreads, E.size[0], E.size[1], E2.value, len);
    // E = E2;

}