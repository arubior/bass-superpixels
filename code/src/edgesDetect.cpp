
#include "edgesDetect.h"

typedef unsigned int uint32;
typedef unsigned short uint16;
typedef unsigned char uint8;

template<typename T>
inline int round(T r){return (int)(r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);}

// construct lookup array for mapping fids to channel indices
uint32* buildLookup( int *dims, int w ) {
    int c, r, z, n=w*w*dims[2]; uint32 *cids=new uint32[n]; n=0;
    for(z=0; z<dims[2]; z++) for(c=0; c<w; c++) for(r=0; r<w; r++)
        cids[n++] = z*dims[0]*dims[1] + c*dims[0] + r;
    return cids;
}

// construct lookup arrays for mapping fids for self-similarity channel
void buildLookupSs( uint32 *&cids1, uint32 *&cids2, int *dims, int w, int m ) {
    int i, j, z, z1, c, r; int locs[1024];
    int m2=m*m, n=m2*(m2-1)/2*dims[2], s=int(w/m/2.0+.5);
    cids1 = new uint32[n]; cids2 = new uint32[n]; n=0;
    for(i=0; i<m; i++) locs[i]=uint32((i+1)*(w+2*s-1)/(m+1.0)-s+.5);
    for(z=0; z<dims[2]; z++) for(i=0; i<m2; i++) for(j=i+1; j<m2; j++) {
        z1=z*dims[0]*dims[1]; n++;
        r=i%m; c=(i-r)/m; cids1[n-1]= z1 + locs[c]*dims[0] + locs[r];
        r=j%m; c=(j-r)/m; cids2[n-1]= z1 + locs[c]*dims[0] + locs[r];
    }
}


void imReshape(_matrix<unsigned char>& src, _matrix<unsigned char>& dst, float resizeScale){

    int dst_size[256];
    for (int i = 0; i < (src.dim-1);i++){
        dst_size[i] = src.size[i] * resizeScale;
    }
    dst_size[src.dim - 1] = src.size[src.dim - 1];
    dst.reShape(src.dim,dst_size);


    float *A1 = new float[src.num];
    float *B1 = new float[dst.num];

    for (int i = 0; i < src.num;i++){
        A1[i] = (float)((unsigned char*)src.value)[i];
    }

    resample(A1, B1, src.size[0], dst.size[0], src.size[1], dst.size[1], dst_size[src.dim - 1], 1.0f);

    for (int i = 0; i < dst.num;i++){
        ((unsigned char*)dst.value)[i] = (unsigned char)(B1[i] + .5);
    }
    delete []A1;
    delete []B1;
}


void imReshape(_matrix<float>& src, _matrix<float>& dst, float resizeScale){
    int dst_size[256];
    for (int i = 0; i < (src.dim - 1); i++){
        dst_size[i] = round(src.size[i] * resizeScale);
    }
    dst_size[src.dim - 1] = src.size[src.dim - 1];
    dst.reShape(src.dim, dst_size);
    resample(src.value,dst.value, src.size[0], dst.size[0], src.size[1], dst.size[1], dst_size[src.dim - 1], 1.0f);
}


void imReshape(_matrix<unsigned char>& src, _matrix<unsigned char>& dst, int dst_h, int dst_w){

    int dst_size[256];
    dst_size[0] = dst_h;
    dst_size[1] = dst_w;
    dst_size[2] = src.size[2];
    dst.reShape(src.dim, dst_size);

    float *A1 = new float[src.num];
    float *B1 = new float[dst.num];

    for (int i = 0; i < src.num; i++){
        A1[i] = (float)((unsigned char*)src.value)[i];
    }

    resample(A1, B1, src.size[0], dst.size[0], src.size[1], dst.size[1], dst_size[src.dim - 1], 1.0f);

    for (int i = 0; i < dst.num; i++){
        ((unsigned char*)dst.value)[i] = (unsigned char)(B1[i] + .5);
    }
    delete[]A1;
    delete[]B1;
}

void imReshape(_matrix<float>& src, _matrix<float>& dst, int dst_h,int dst_w){
    int dst_size[256];
    dst_size[0] = dst_h;
    dst_size[1] = dst_w;
    dst_size[2] = src.size[2];
    dst.reShape(src.dim, dst_size);
    resample(src.value, dst.value, src.size[0], dst.size[0], src.size[1], dst.size[1], dst_size[2], 1.0f);
}


void imPad(_matrix<unsigned char>& src, _matrix<unsigned char>& dst, int* p,int flag){

    int pt, pb, pl, pr;
    pt = int(p[0]); pb = int(p[1]); pl = int(p[2]); pr = int(p[3]);

    int ns[3], ms[3];
    ns[0] = src.size[0];
    ns[1] = src.size[1];
    ns[2] = src.size[2];

    ms[0] = ns[0] + pt + pb; ms[1] = ns[1] + pl + pr; ms[2] = ns[2];
    if (ms[0] < 0 || ns[0] <= -pt || ns[0] <= -pb) ms[0] = 0;
    if (ms[1] < 0 || ns[1] <= -pl || ns[1] <= -pr) ms[1] = 0;

    dst.reShape(src.dim, ms);

    unsigned char val = 0;
    imPad(src.value, dst.value, ns[0], ns[1], ns[2], pt, pb, pl, pr, flag, val);
}

void rgbConvert(_matrix<unsigned char>& src,_matrix<float>& dst,int flag,int useSingle, const bool useSSE)
{
    int n = src.size[0] * src.size[1];
    int d = src.size[2];
    rgbConvert(src.value, dst.value, n, d, flag, 1.0f/255, useSSE);
}

void convTri(_matrix<float>& src, _matrix<float>& dst, int grdSmooth)
{
    dst.reShape(src.dim, src.size);
    dst.setValue(0.0f);
    if (grdSmooth == 0){
        dst = src;
    }
    else{
        if (grdSmooth > 0 && grdSmooth <= 1){
            convTri1(src.value, dst.value, src.size[0], src.size[1], src.size[2], 12 / grdSmooth / (grdSmooth+2) -2, 1);
        }
        else{
            //convTri(A, B, ns[0], ns[1], d, r, s);
            convTri(src.value, dst.value, src.size[0], src.size[1], src.size[2], grdSmooth, 1);
        }
    }
}

void gradientMag(_matrix<float>& I, _matrix<float>& M, _matrix<float>& O, int channel, int normRad, float normConst){
    int size[3];
    size[0] = I.size[0];
    size[1] = I.size[1];
    size[2] = 1;

    M.reShape(I.dim, size);
    O.reShape(I.dim, size);
    gradMag(I.value, M.value, O.value, I.size[0], I.size[1], I.size[2],0);
    _matrix<float> S;
    S.reShape(I.dim, I.size);
    convTri(M, S, normRad);
    gradMagNorm(M.value, S.value, M.size[0], M.size[1], normConst);
}


void gradientHist(_matrix<float>& M, _matrix<float>& O, _matrix<float>& H, int binSize, int nOrients)
{
    int size[3];
    size[0] = M.size[0]/binSize;
    size[1] = M.size[1]/binSize;
    size[2] = 4;
    H.reShape(M.dim,size);
    int softBin = 0; bool full = false;
    H.setValue(0);
    gradHist(M.value, O.value, H.value, M.size[0], M.size[1], binSize, nOrients, softBin, full);
}

void edgesChns(_matrix<unsigned char>& I, Model& model, _matrix<float>& chnsReg, _matrix<float>& chnsSim)
{
    int shrink = model.opts.shrink;
    int nTypes = 1; int k = 0;

    _matrix<float>* chns = new _matrix<float>[model.opts.nChns];

    _matrix<float> It;
    It.reShape(I.dim,I.size);
    rgbConvert(I,It,2,1);

    _matrix<float> Ishrink;

    float resizeScale = 1.0f / shrink;
    imReshape(It, Ishrink, resizeScale);
    chns[k] = Ishrink;
    k = k + 1;

    for (int i = 1; i <= 2;i++){
        int s = pow(double(2),i-1);
        _matrix<float> I1;
        if (s==shrink){
            I1 = Ishrink;
        }
        else{
            if (s==1){
                I1 = It;
            }
            else{
                imReshape(It,I1,1.0f / (float)s);
            }
        }
        _matrix<float> I2;
        convTri(I1,I2,model.opts.grdSmooth);

        _matrix<float> M;  _matrix<float> O;
        gradientMag(I2, M, O, 0, model.opts.normRad, 0.01f);

        _matrix<float> H;
        int binSize = 1 > ((float)shrink / (float)s) ? 1 : ((float)shrink / (float)s);
        gradientHist(M, O, H, binSize, model.opts.nOrients);

        imReshape(M, chns[k],(float)s/(float)shrink); k = k + 1;
        imReshape(H, chns[k], 1.0 > ((float)s / (float)shrink) ? 1 : ((float)s / (float)shrink)); k = k + 1;
    }

    _matrix<float> chns_t;
    chns_t.cat3(chns,k);

    int chnSm = model.opts.chnSmooth / shrink;
    int simSm = model.opts.simSmooth / shrink;

    convTri(chns_t, chnsReg, chnSm);
    convTri(chns_t, chnsSim, simSm);

    delete []chns;
}

//void edgesDetect(Model& model, float* I, float* chns, float *chnsSs, int* sizeImg, _matrix<float>& E_m, _matrix<int>& ind_m)


// [E,ind,segs] = mexFunction(model,I,chns,chnsSs) - helper for edgesDetect.m
//void edgesDetect(Model& model,float* I, float* chns,float *chnsSs,int* sizeImg,float* E, int* ind)
void edgesDetect(Model& model, float* I, float* chns, float *chnsSs, int* sizeImg, _matrix<float>& E_m, _matrix<int>& ind_m)
{
    float *thrs = model.thrs.value;
    int *fids = model.fids.value;
    int* child = model.child.value;
    unsigned char* segs = model.segs.value;
    int* nSegs = model.nSegs.value;
    int* eBins = model.eBins.value;
    int *eBnds = model.eBnds.value;

    const int shrink = model.opts.shrink;
    const int imWidth = model.opts.imWidth;
    const int gtWidth = model.opts.gtWidth;
    const int nChns = model.opts.nChns;
    const int nCells = model.opts.nCells;
    const int nChnFtrs = model.opts.nChnFtrs;
    const int stride = model.opts.stride;
    const int nTreesEval = model.opts.nTreesEval;
    int sharpen = model.opts.sharpen;
    int nThreads = model.opts.nThreads;

    const int nBnds = (model.eBnds.num - 1) / (model.thrs.num - 1);
    int h = sizeImg[0];
    int w = sizeImg[1];
    int Z = sizeImg[2];
    const int nTreeNodes = model.fids.size[0];
    const int nTrees = model.fids.size[1];

    const int h1 = (int)ceil(double(h - imWidth) / stride);
    const int w1 = (int)ceil(double(w - imWidth) / stride);
    const int h2 = h1*stride + gtWidth;
    const int w2 = w1*stride + gtWidth;
    const int imgDims[3] = { h, w, Z };
    const int chnDims[3] = { h / shrink, w / shrink, nChns };
    int indDims[3] = { h1, w1, nTreesEval };
    int outDims[3] = { h2, w2, 1 };
    const int segDims[5] = { gtWidth, gtWidth, h1, w1, nTreesEval };
    uint32 *iids, *eids, *cids, *cids1, *cids2;
    iids = buildLookup((int*)imgDims, gtWidth);
    eids = buildLookup((int*)outDims, gtWidth);
    cids = buildLookup((int*)chnDims, imWidth / shrink);
    buildLookupSs(cids1, cids2, (int*)chnDims, imWidth / shrink, nCells);

    E_m.reShape(3, outDims);
    E_m.setValue(0);
    ind_m.reShape(3, indDims);
    int* ind = ind_m.value;

#if USEOMP
    nThreads = nThreads < omp_get_max_threads() ? nThreads : omp_get_max_threads();
#pragma omp parallel for num_threads(nThreads)
#endif
    for (int c = 0; c < w1; c++)
        for (int t = 0; t < nTreesEval; t++) {
            for (int r0 = 0; r0 < 2; r0++){

                for (int r = r0; r < h1; r += 2) {
                    int o = (r*stride / shrink) + (c*stride / shrink)*h / shrink;
                    // select tree to evaluate
                    int t1 = ((r + c) % 2 * nTreesEval + t) % nTrees; uint32 k = t1*nTreeNodes;
                    while (child[k]) {
                        // compute feature (either channel or self-similarity feature)
                        uint32 f = fids[k]; float ftr;
                        if (f < nChnFtrs)
                            ftr = chns[cids[f] + o];
                        else
                            ftr = chnsSs[cids1[f - nChnFtrs] + o] - chnsSs[cids2[f - nChnFtrs] + o];
                        // compare ftr to threshold and move left or right accordingly
                        if (ftr < thrs[k]){
                            k = child[k] - 1;
                        }
                        else{
                            k = child[k];
                        }
                        k += t1*nTreeNodes;
                    }
                    // store leaf index and update edge maps
                    ind[r + c*h1 + t*h1*w1] = k;
                }
            }
        }

    // compute edge maps (avoiding collisions from parallel executions)
    if (!sharpen) for (int c0 = 0; c0 < gtWidth / stride; c0++) {
#if USEOMP
#pragma omp parallel for num_threads(nThreads)
#endif
        for (int c = c0; c < w1; c += gtWidth / stride) {
            for (int r = 0; r < h1; r++) for (int t = 0; t < nTreesEval; t++) {
                uint32 k = ind[r + c*h1 + t*h1*w1];
                float *E1 = E_m.value + (r*stride) + (c*stride)*h2;
                int b0 = eBnds[k*nBnds], b1 = eBnds[k*nBnds + 1]; if (b0 == b1) continue;
                for (int b = b0; b < b1; b++) E1[eids[eBins[b]]]++;
                // 				if(nl>2) memcpy(segsOut+(r+c*h1+t*h1*w1)*gtWidth*gtWidth,
                // 					segs+k*gtWidth*gtWidth,gtWidth*gtWidth*sizeof(uint8));
            }
        }
    }

    // computed sharpened edge maps, snapping to local color values
    if (sharpen) {
        // compute neighbors array
        const int g = gtWidth; uint16 N[4096 * 4];
        for (int c = 0; c < g; c++) for (int r = 0; r<g; r++) {
            int i = c*g + r; uint16 *N1 = N + i * 4;
            N1[0] = c>0 ? i - g : i; N1[1] = c<g - 1 ? i + g : i;
            N1[2] = r>0 ? i - 1 : i; N1[3] = r < g - 1 ? i + 1 : i;
        }
#if USEOMP
#pragma omp parallel for num_threads(nThreads)
#endif
        for (int c = 0; c < w1; c++) for (int r = 0; r < h1; r++) {
            for (int t = 0; t < nTreesEval; t++) {
                // get current segment and copy into S
                uint32 k = ind[r + c*h1 + t*h1*w1];
                int m = nSegs[k]; if (m == 1) continue;
                //uint8 S0[4096], *S=(nl<=2) ? S0 : segsOut+(r+c*h1+t*h1*w1)*g*g;
                uint8 S0[4096], *S = S0;
                memcpy(S, segs + k*g*g, g*g*sizeof(uint8));
                // compute color model for each segment using every other pixel
                int ci, ri, s, z; float ns[100], mus[1000];
                const float *I1 = I + (c*stride + (imWidth - g) / 2)*h + r*stride + (imWidth - g) / 2;
                for (s = 0; s < m; s++) { ns[s] = 0; for (z = 0; z < Z; z++) mus[s*Z + z] = 0; }
                for (ci = 0; ci < g; ci += 2) for (ri = 0; ri < g; ri += 2) {
                    s = S[ci*g + ri]; ns[s]++;
                    for (z = 0; z < Z; z++) mus[s*Z + z] += I1[z*h*w + ci*h + ri];
                }
                for (s = 0; s < m; s++) for (z = 0; z < Z; z++) mus[s*Z + z] /= ns[s];
                int b0 = eBnds[k*nBnds], b1 = eBnds[k*nBnds + sharpen];
                for (int b = b0; b < b1; b++) {
                    float vs[10], d, e, eBest = 1e10f; int i, sBest = -1, ss[4];
                    for (i = 0; i < 4; i++) ss[i] = S[N[eBins[b] * 4 + i]];
                    for (z = 0; z < Z; z++) vs[z] = I1[iids[eBins[b]] + z*h*w];
                    for (i = 0; i < 4; i++) {
                        s = ss[i]; if (s == sBest) continue;
                        e = 0; for (z = 0; z < Z; z++) { d = mus[s*Z + z] - vs[z]; e += d*d; }
                        if (e < eBest) { eBest = e; sBest = s; }
                    }
                    S[eBins[b]] = sBest;
                }
                // convert mask to edge maps (examining expanded set of pixels)
                float *E1 = E_m.value + c*stride*h2 + r*stride; b1 = eBnds[k*nBnds + sharpen + 1];
                for (int b = b0; b < b1; b++) {
                    int i = eBins[b]; uint8 s = S[i]; uint16 *N1 = N + i * 4;
                    if (s != S[N1[0]] || s != S[N1[1]] || s != S[N1[2]] || s != S[N1[3]])
                        E1[eids[i]]++;
                }
            }
        }
    }
    delete[] iids;
    delete[] eids;
    delete[] cids;
    delete[] cids1;
    delete[] cids2;
}


void edgesDetect(Model& model, _matrix<float>& I, _matrix<float>& chnsReg, _matrix<float>& chnsSim, _matrix<float>& E, _matrix<int>& inds){
    //edgesDetect(model, I.value, chnsReg.value, chnsSim.value, I.size, E, inds);
    edgesDetect(model, I.value, chnsReg.value, chnsSim.value, I.size, E, inds);
}
/*
void multipleValue(_matrix<float>& mat, int startY,int endY,int startX,int endX,float t)
{
    // suppose the dim is 3.
    int height = mat.size[0];
    int width = mat.size[1];
    int channel = mat.size[2];

    int ind1 = 0;
    for (int k = 0; k < channel;k++)
    {
        int ind2 = ind1+startX*height;
        for (int j = startX; j < width - endX;j++)
        {
            for (int i = startY; i < height - endY;i++)
            {
                mat.value[ind2+i] *= t;
            }
            ind2 += height;
        }
        ind1 += height*width;
    }
}
*/

void multipleValue(_matrix<float>& mat, int startY, int endY, int startX, int endX, float t)
{
    // suppose the dim is 3.
    int height = mat.size[0];
    int width = mat.size[1];
    int channel = mat.size[2];

    _matrix<float> dst;

    int size[3] = { endY - startY + 1, endX - startX + 1, channel };
    dst.reShape(3,size);
    dst.setValue(0);
    int ind1 = 0;

    int ind1_dst = 0;
    for (int k = 0; k < channel; k++)
    {
        int ind2 = ind1 + startX*height;
        int ind2_dst = ind1_dst;
        for (int j = startX; j < endX; j++)
        {
            for (int i = startY; i < endY; i++)
            {
                dst.value[ind2_dst + i - startY] = mat.value[ind2 + i] * t;
            }
            ind2 += height;
            ind2_dst += size[0];
        }
        ind1 += height*width;
        ind1_dst += size[0]*size[1];
    }
    mat = dst;
}


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

void calcO(_matrix<float>& E,_matrix<float>& O)
{
    _matrix<float> E2;
    convTri(E, E2, 4);

    _matrix<float> Ox;
    _matrix<float> Oy;
    Ox.reShape(E2.dim,E2.size);
    Oy.reShape(E2.dim, E2.size);
    Ox.setValue(0);
    Oy.setValue(0);

    grad2(E2.value, Ox.value, Oy.value, E2.size[0], E2.size[1], E2.size[2]);


    _matrix<float> Oxx;
    _matrix<float> O_tmp;
    Oxx.reShape(E2.dim, E2.size);
    O_tmp.reShape(E2.dim, E2.size);
    Oxx.setValue(0);
    O_tmp.setValue(0);
    grad2(Ox.value, Oxx.value, O_tmp.value, Ox.size[0], Ox.size[1], Ox.size[2]);

    _matrix<float> Oxy;
    _matrix<float> Oyy;
    Oxy.reShape(E2.dim, E2.size);
    Oyy.reShape(E2.dim, E2.size);
    Oxy.setValue(0);
    Oyy.setValue(0);
    grad2(Oy.value, Oxy.value, Oyy.value, Ox.size[0], Ox.size[1], Ox.size[2]);

    O.reShape(E2.dim, E2.size);

    for (int i = 0; i < O.num;i++){
        float t = atan(Oyy.value[i] * sgn(-Oxy.value[i]) / (Oxx.value[i] + 1e-32f));
        O.value[i] = fmod( t,PI);
    }

}

void edgesDetect(_matrix<unsigned char>& I, Model& model, _matrix<float>& E, _matrix<float>& O)
{


    int sizeE[3];
    sizeE[0] = I.size[0]; sizeE[1] = I.size[1]; sizeE[2] = 1;
    E.reShape(I.dim, sizeE);
    E.setValue(0);

    if (model.opts.multiscale == 1){
        float ss[3] = {0.5,1.0f,2.0f};
        model.opts.multiscale = 0;

        _matrix<float> E1[3];
        _matrix<float> O1[3];
        for (int k = 0; k < 3; k++){
            _matrix<unsigned char> I1;
            imReshape(I,I1,ss[k]);
            E1[k].reShape(I1.dim, I1.size);
            O1[k].reShape(I1.dim, I1.size);
            edgesDetect(I1, model, E1[k], O1[k]);

            _matrix<float> tmpE;
            imReshape(E1[k], tmpE, I.size[0], I.size[1]);

            for (int i = 0; i < E.num;i++){
                E.value[i] += tmpE.value[i];
            }
        }

        for (int i = 0; i < E.num; i++){
            E.value[i] /= 3;
        }
    }
    else{
        int r = model.opts.imWidth / 2;
        int p[4] = {r,r,r,r};
        p[1] = p[1] + (4 - (I.size[0] + 2 * r) % 4) % 4;
        p[3] = p[3] + (4 - (I.size[1] + 2 * r) % 4) % 4;

        _matrix<unsigned char> It;
        imPad(I,It,p,2);

        _matrix<float>chnsReg;
        _matrix<float>chnsSim;
        edgesChns(It, model, chnsReg, chnsSim);

        int s = model.opts.sharpen;
        if (s>0){
            _matrix<float> I1;
            I1.reShape(It.dim, It.size);
            rgbConvert(It, I1, 1, 1);
            _matrix<float> I2;
            convTri(I1, I2, 1);

            _matrix<int> ind;
            edgesDetect(model, I2, chnsReg, chnsSim, E, ind);

            float t = (float)(model.opts.stride*model.opts.stride) / (float)model.opts.nTreesEval / (float)(model.opts.gtWidth*model.opts.gtWidth);
            int r = model.opts.gtWidth / 2;
            if (s == 0)
                t = t * 2;
            else if(s == 1)
                t = t*1.8;
            else
                t = t*1.66;

            multipleValue(E, r, r - 1 + I.size[0], r, r - 1 + I.size[1], t);
            _matrix<float> E1;
            convTri(E, E1, 1);
            E = E1;
        }
    }

    calcO(E, O);

}
