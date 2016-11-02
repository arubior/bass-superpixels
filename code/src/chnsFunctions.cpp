#include "chnsFunctions.h"
#include "config.h"
#include <string.h>

// platform independent aligned memory allocation (see also alFree)


/*
////////////////////////////////////////////////////////////////////////////////////////////
///  rgbConvertMex.cpp
////////////////////////////////////////////////////////////////////////////////////////////
// Constants for rgb2luv conversion and lookup table for y-> l conversion
template<class oT> oT* rgb2luv_setup(oT z, oT *mr, oT *mg, oT *mb,
    oT &minu, oT &minv, oT &un, oT &vn)
{
    // set constants for conversion
    const oT y0 = (oT)((6.0 / 29)*(6.0 / 29)*(6.0 / 29));
    const oT a = (oT)((29.0 / 3)*(29.0 / 3)*(29.0 / 3));
    un = (oT) 0.197833; vn = (oT) 0.468331;
    mr[0] = (oT) 0.430574*z; mr[1] = (oT) 0.222015*z; mr[2] = (oT) 0.020183*z;
    mg[0] = (oT) 0.341550*z; mg[1] = (oT) 0.706655*z; mg[2] = (oT) 0.129553*z;
    mb[0] = (oT) 0.178325*z; mb[1] = (oT) 0.071330*z; mb[2] = (oT) 0.939180*z;
    oT maxi = (oT) 1.0 / 270; minu = -88 * maxi; minv = -134 * maxi;
    // build (padded) lookup table for y->l conversion assuming y in [0,1]
    static oT lTable[1064]; static bool lInit = false;
    if (lInit) return lTable; oT y, l;
    for (int i = 0; i<1025; i++) {
        y = (oT)(i / 1024.0);
        l = y>y0 ? 116 * (oT)pow((double)y, 1.0 / 3.0) - 16 : y*a;
        lTable[i] = l*maxi;
    }
    for (int i = 1025; i < 1064; i++) lTable[i] = lTable[i - 1];
    lInit = true; return lTable;
}

// Convert from rgb to luv
template<class iT, class oT> void rgb2luv(iT *I, oT *J, int n, oT nrm)
{
    oT minu, minv, un, vn, mr[3], mg[3], mb[3];
    oT *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
    oT *L = J, *U = L + n, *V = U + n; iT *R = I, *G = R + n, *B = G + n;
    for (int i = 0; i < n; i++) {
        oT r, g, b, x, y, z, l;
        r = (oT)*R++; g = (oT)*G++; b = (oT)*B++;
        x = mr[0] * r + mg[0] * g + mb[0] * b;
        y = mr[1] * r + mg[1] * g + mb[1] * b;
        z = mr[2] * r + mg[2] * g + mb[2] * b;
        l = lTable[(int)(y * 1024)];
        *(L++) = l; z = 1 / (x + 15 * y + 3 * z + (oT)1e-35);
        *(U++) = l * (13 * 4 * x*z - 13 * un) - minu;
        *(V++) = l * (13 * 9 * y*z - 13 * vn) - minv;
    }
}

// Convert from rgb to luv using sse
template<class iT> void rgb2luv_sse(iT *I, float *J, int n, float nrm) {
    const int k = 256; float R[k], G[k], B[k];
    if ((size_t(R) & 15 || size_t(G) & 15 || size_t(B) & 15 || size_t(I) & 15 || size_t(J) & 15)
        || n % 4 > 0) {
        rgb2luv(I, J, n, nrm); return;
    }
    int i = 0, i1, n1; float minu, minv, un, vn, mr[3], mg[3], mb[3];
    float *lTable = rgb2luv_setup(nrm, mr, mg, mb, minu, minv, un, vn);
    while (i<n) {
        n1 = i + k; if (n1>n) n1 = n; float *J1 = J + i; float *R1, *G1, *B1;
        // convert to floats (and load input into cache)
        if (typeid(iT) != typeid(float)) {
            R1 = R; G1 = G; B1 = B; iT *Ri = I + i, *Gi = Ri + n, *Bi = Gi + n;
            for (i1 = 0; i1 < (n1 - i); i1++) {
                R1[i1] = (float)*Ri++; G1[i1] = (float)*Gi++; B1[i1] = (float)*Bi++;
            }
        }
        else { R1 = ((float*)I) + i; G1 = R1 + n; B1 = G1 + n; }
        // compute RGB -> XYZ
        for (int j = 0; j < 3; j++) {
            __m128 _mr, _mg, _mb, *_J = (__m128*) (J1 + j*n);
            __m128 *_R = (__m128*) R1, *_G = (__m128*) G1, *_B = (__m128*) B1;
            _mr = SET(mr[j]); _mg = SET(mg[j]); _mb = SET(mb[j]);
            for (i1 = i; i1 < n1; i1 += 4) *(_J++) = ADD(ADD(MUL(*(_R++), _mr),
                MUL(*(_G++), _mg)), MUL(*(_B++), _mb));
        }
        { // compute XZY -> LUV (without doing L lookup/normalization)
            __m128 _c15, _c3, _cEps, _c52, _c117, _c1024, _cun, _cvn;
            _c15 = SET(15.0f); _c3 = SET(3.0f); _cEps = SET(1e-35f);
            _c52 = SET(52.0f); _c117 = SET(117.0f), _c1024 = SET(1024.0f);
            _cun = SET(13 * un); _cvn = SET(13 * vn);
            __m128 *_X, *_Y, *_Z, _x, _y, _z;
            _X = (__m128*) J1; _Y = (__m128*) (J1 + n); _Z = (__m128*) (J1 + 2 * n);
            for (i1 = i; i1 < n1; i1 += 4) {
                _x = *_X; _y = *_Y; _z = *_Z;
                _z = RCP(ADD(_x, ADD(_cEps, ADD(MUL(_c15, _y), MUL(_c3, _z)))));
                *(_X++) = MUL(_c1024, _y);
                *(_Y++) = SUB(MUL(MUL(_c52, _x), _z), _cun);
                *(_Z++) = SUB(MUL(MUL(_c117, _y), _z), _cvn);
            }
        }
        { // perform lookup for L and finalize computation of U and V
            for (i1 = i; i1 < n1; i1++) J[i1] = lTable[(int)J[i1]];
            __m128 *_L, *_U, *_V, _l, _cminu, _cminv;
            _L = (__m128*) J1; _U = (__m128*) (J1 + n); _V = (__m128*) (J1 + 2 * n);
            _cminu = SET(minu); _cminv = SET(minv);
            for (i1 = i; i1 < n1; i1 += 4) {
                _l = *(_L++);
                *(_U++) = SUB(MUL(_l, *_U), _cminu);
                *(_V++) = SUB(MUL(_l, *_V), _cminv);
            }
        }
        i = n1;
    }
}

// Convert from rgb to hsv
template<class iT, class oT> void rgb2hsv(iT *I, oT *J, int n, oT nrm) {
    oT *H = J, *S = H + n, *V = S + n;
    iT *R = I, *G = R + n, *B = G + n;
    for (int i = 0; i < n; i++) {
        const oT r = (oT)*(R++), g = (oT)*(G++), b = (oT)*(B++);
        oT h, s, v, minv, maxv;
        if (r == g && g == b) {
            *(H++) = 0; *(S++) = 0; *(V++) = r*nrm; continue;
        }
        else if (r >= g && r >= b) {
            maxv = r; minv = g < b ? g : b;
            h = (g - b) / (maxv - minv) + 6; if (h >= 6) h -= 6;
        }
        else if (g >= r && g >= b) {
            maxv = g; minv = r < b ? r : b;
            h = (b - r) / (maxv - minv) + 2;
        }
        else {
            maxv = b; minv = r < g ? r : g;
            h = (r - g) / (maxv - minv) + 4;
        }
        h *= (oT)(1 / 6.0); s = 1 - minv / maxv; v = maxv*nrm;
        *(H++) = h; *(S++) = s; *(V++) = v;
    }
}

// Convert from rgb to gray
template<class iT, class oT> void rgb2gray(iT *I, oT *J, int n, oT nrm) {
    oT *GR = J; iT *R = I, *G = R + n, *B = G + n; int i;
    oT mr = (oT).2989360213*nrm, mg = (oT).5870430745*nrm, mb = (oT).1140209043*nrm;
    for (i = 0; i < n; i++) *(GR++) = (oT)*(R++)*mr + (oT)*(G++)*mg + (oT)*(B++)*mb;
}

// Convert from rgb (double) to gray (float)
template<> void rgb2gray(double *I, float *J, int n, float nrm) {
    float *GR = J; double *R = I, *G = R + n, *B = G + n; int i;
    double mr = .2989360213*nrm, mg = .5870430745*nrm, mb = .1140209043*nrm;
    for (i = 0; i < n; i++) *(GR++) = (float)(*(R++)*mr + *(G++)*mg + *(B++)*mb);
}

// Copy and normalize only
template<class iT, class oT> void normalize(iT *I, oT *J, int n, oT nrm) {
    for (int i = 0; i < n; i++) *(J++) = (oT)*(I++)*nrm;
}

// Convert rgb to various colorspaces
template<class iT, class oT>
oT* rgbConvert(iT *I, int n, int d, int flag, oT nrm) {
    oT *J = (oT*)wrMalloc(n*(flag == 0 ? 1 : d)*sizeof(oT));
    int n1 = d*(n<1000 ? n / 10 : 100); oT thr = oT(1.001);
    if (flag>1 && nrm == 1) for (int i = 0; i<n1; i++) if (I[i]>thr)
        wrError("For floats all values in I must be smaller than 1.");
    bool useSse = n % 4 == 0 && typeid(oT) == typeid(float);
    if (flag == 2 && useSse) rgb2luv_sse(I, (float*)J, n, (float)nrm);
    else if (flag == 0 && d == 1) normalize(I, J, n, nrm);
    else if (flag == 0) rgb2gray(I, J, n, nrm);
    else if (flag == 1) normalize(I, J, n*d, nrm);
    else if (flag == 2) rgb2luv(I, J, n, nrm);
    else if (flag == 3) rgb2hsv(I, J, n, nrm);
    else wrError("Unknown flag.");
    return J;
}
*/


////////////////////////////////////////////////////////////////////////////////////////////
//   convConst.cpp
////////////////////////////////////////////////////////////////////////////////////////////

// convolve two columns of I by ones filter
void convBoxY(float *I, float *O, int h, int r, int s) {
    float t; int j, p = r + 1, q = 2 * h - (r + 1), h0 = r + 1, h1 = h - r, h2 = h;
    t = 0; for (j = 0; j <= r; j++) t += I[j]; t = 2 * t - I[r]; j = 0;
    if (s == 1) {
        for (; j < h0; j++) O[j] = t -= I[r - j] - I[r + j];
        for (; j<h1; j++) O[j] = t -= I[j - p] - I[r + j];
        for (; j<h2; j++) O[j] = t -= I[j - p] - I[q - j];
    }
    else {
        int k = (s - 1) / 2; h2 = (h / s)*s; if (h0>h2) h0 = h2; if (h1>h2) h1 = h2;
        for (; j < h0; j++) { t -= I[r - j] - I[r + j]; k++; if (k == s) { k = 0; *O++ = t; } }
        for (; j < h1; j++) { t -= I[j - p] - I[r + j]; k++; if (k == s) { k = 0; *O++ = t; } }
        for (; j<h2; j++) { t -= I[j - p] - I[q - j]; k++; if (k == s) { k = 0; *O++ = t; } }
    }
}

// convolve I by a 2r+1 x 2r+1 ones filter (uses SSE)
void convBox(float *I, float *O, int h, int w, int d, int r, int s) {
    float nrm = 1.0f / ((2 * r + 1)*(2 * r + 1)); int i, j, k = (s - 1) / 2, h0, h1, w0;
    if (h % 4 == 0) h0 = h1 = h; else { h0 = h - (h % 4); h1 = h0 + 4; } w0 = (w / s)*s;
    float *T = (float*)alMalloc(h1*sizeof(float), 16);
    while (d-- > 0) {
        // initialize T
        memset(T, 0, h1*sizeof(float));
        for (i = 0; i <= r; i++) for (j = 0; j < h0; j += 4) INC(T[j], LDu(I[j + i*h]));
        for (j = 0; j < h0; j += 4) STR(T[j], MUL(nrm, SUB(MUL(2, LD(T[j])), LDu(I[j + r*h]))));
        for (i = 0; i <= r; i++) for (j = h0; j < h; j++) T[j] += I[j + i*h];
        for (j = h0; j < h; j++) T[j] = nrm*(2 * T[j] - I[j + r*h]);
        // prepare and convolve each column in turn
        for (i = 0; i < w0; i++) {
            float *Il, *Ir; Il = Ir = I;
            if (i <= r) { Il += (r - i)*h; Ir += (r + i)*h; }
            else if (i < w - r) { Il -= (r - i + 1)*h; Ir += (r + i)*h; }
            else { Il -= (r - i + 1)*h; Ir += 2 * w*h - (r + i + 1)*h; }
            if (i) for (j = 0; j < h0; j += 4) DEC(T[j], MUL(nrm, SUB(LDu(Il[j]), LDu(Ir[j]))));
            if (i) for (j = h0; j < h; j++) T[j] -= nrm*(Il[j] - Ir[j]);
            k++; if (k == s) { k = 0; convBoxY(T, O, h, r, s); O += h / s; }
        }
        I += w*h;
    }
    alFree(T);
}

// convolve single column of I by [1; 1] filter (uses SSE)
void conv11Y(float *I, float *O, int h, int side, int s) {
#define C4(m,o) ADD(LDu(I[m*j-1+o]),LDu(I[m*j+o]))
    int j = 0, k = ((~((size_t)O) + 1) & 15) / 4;
    const int d = (side % 4 >= 2) ? 1 : 0, h2 = (h - d) / 2;
    if (s == 2) {
        for (; j < k; j++) O[j] = I[2 * j + d] + I[2 * j + d + 1];
        for (; j < h2 - 4; j += 4) STR(O[j], _mm_shuffle_ps(C4(2, d + 1), C4(2, d + 5), 136));
        for (; j < h2; j++) O[j] = I[2 * j + d] + I[2 * j + d + 1];
        if (d == 1 && h % 2 == 0) O[j] = 2 * I[2 * j + d];
    }
    else {
        if (d == 0) { O[0] = 2 * I[0]; j++; if (k == 0) k = 4; }
        for (; j < k; j++) O[j] = I[j - 1 + d] + I[j + d];
        for (; j < h - 4 - d; j += 4) STR(O[j], C4(1, d));
        for (; j < h - d; j++) O[j] = I[j - 1 + d] + I[j + d];
        if (d == 1) { O[j] = 2 * I[j]; j++; }
    }
#undef C4
}

// convolve I by [1 1; 1 1] filter (uses SSE)
void conv11(float *I, float *O, int h, int w, int d, int side, int s) {
    const float nrm = 0.25f; int i, j;
    float *I0, *I1, *T = (float*)alMalloc(h*sizeof(float), 16);
    for (int d0 = 0; d0 < d; d0++) for (i = s / 2; i < w; i += s) {
        I0 = I1 = I + i*h + d0*h*w; if (side % 2) { if (i < w - 1) I1 += h; }
        else { if (i) I0 -= h; }
        for (j = 0; j < h - 4; j += 4) STR(T[j], MUL(nrm, ADD(LDu(I0[j]), LDu(I1[j]))));
        for (; j < h; j++) T[j] = nrm*(I0[j] + I1[j]);
        conv11Y(T, O, h, side, s); O += h / s;
    }
    alFree(T);
}

// convolve one column of I by a 2rx1 triangle filter
void convTriY(float *I, float *O, int h, int r, int s) {
    r++; float t, u; int j, r0 = r - 1, r1 = r + 1, r2 = 2 * h - r, h0 = r + 1, h1 = h - r + 1, h2 = h;
    u = t = I[0]; for (j = 1; j < r; j++) u += t += I[j]; u = 2 * u - t; t = 0;
    if (s == 1) {
        O[0] = u; j = 1;
        for (; j < h0; j++) O[j] = u += t += I[r - j] + I[r0 + j] - 2 * I[j - 1];
        for (; j<h1; j++) O[j] = u += t += I[j - r1] + I[r0 + j] - 2 * I[j - 1];
        for (; j<h2; j++) O[j] = u += t += I[j - r1] + I[r2 - j] - 2 * I[j - 1];
    }
    else {
        int k = (s - 1) / 2; h2 = (h / s)*s; if (h0>h2) h0 = h2; if (h1>h2) h1 = h2;
        if (++k == s) { k = 0; *O++ = u; } j = 1;
        for (; j < h0; j++) { u += t += I[r - j] + I[r0 + j] - 2 * I[j - 1]; if (++k == s){ k = 0; *O++ = u; } }
        for (; j < h1; j++) { u += t += I[j - r1] + I[r0 + j] - 2 * I[j - 1]; if (++k == s){ k = 0; *O++ = u; } }
        for (; j<h2; j++) { u += t += I[j - r1] + I[r2 - j] - 2 * I[j - 1]; if (++k == s){ k = 0; *O++ = u; } }
    }
}

// convolve I by a 2rx1 triangle filter (uses SSE)
void convTri(float *I, float *O, int h, int w, int d, int r, int s) {
    r++; float nrm = 1.0f / (r*r*r*r); int i, j, k = (s - 1) / 2, h0, h1, w0;
    if (h % 4 == 0) h0 = h1 = h; else { h0 = h - (h % 4); h1 = h0 + 4; } w0 = (w / s)*s;
    float *T = (float*)alMalloc(2 * h1*sizeof(float), 16), *U = T + h1;
    while (d-- > 0) {
        // initialize T and U
        for (j = 0; j < h0; j += 4) STR(U[j], STR(T[j], LDu(I[j])));
        for (i = 1; i < r; i++) for (j = 0; j < h0; j += 4) INC(U[j], INC(T[j], LDu(I[j + i*h])));
        for (j = 0; j < h0; j += 4) STR(U[j], MUL(nrm, (SUB(MUL(2, LD(U[j])), LD(T[j])))));
        for (j = 0; j < h0; j += 4) STR(T[j], 0);
        for (j = h0; j < h; j++) U[j] = T[j] = I[j];
        for (i = 1; i < r; i++) for (j = h0; j < h; j++) U[j] += T[j] += I[j + i*h];
        for (j = h0; j < h; j++) { U[j] = nrm * (2 * U[j] - T[j]); T[j] = 0; }
        // prepare and convolve each column in turn
        for (i = 0; i < w0; i++) {
            float *Il, *Ir, *Im; Il = Ir = Im = I; Im += (i - 1)*h;
            if (i <= r) { Il += (r - i)*h; Ir += (r - 1 + i)*h; }
            else if (i <= w - r) { Il -= (r + 1 - i)*h; Ir += (r - 1 + i)*h; }
            else { Il -= (r + 1 - i)*h; Ir += (2 * w - r - i)*h; }
            if (i) for (j = 0; j < h0; j += 4) {
                __m128 del = SUB(ADD(LDu(Il[j]), LDu(Ir[j])), MUL(2, LDu(Im[j])));
                INC(U[j], MUL(nrm, (INC(T[j], del))));
            }
            if (i) for (j = h0; j < h; j++) U[j] += nrm*(T[j] += Il[j] + Ir[j] - 2 * Im[j]);
            k++; if (k == s) { k = 0; convTriY(U, O, h, r - 1, s); O += h / s; }
        }
        I += w*h;
    }
    alFree(T);
}

// convolve one column of I by [1 p 1] filter (uses SSE)
void convTri1Y(float *I, float *O, int h, float p, int s) {
#define C4(m,o) ADD(ADD(LDu(I[m*j-1+o]),MUL(p,LDu(I[m*j+o]))),LDu(I[m*j+1+o]))
    int j = 0, k = ((~((size_t)O) + 1) & 15) / 4, h2 = (h - 1) / 2;
    if (s == 2) {
        for (; j < k; j++) O[j] = I[2 * j] + p*I[2 * j + 1] + I[2 * j + 2];
        for (; j < h2 - 4; j += 4) STR(O[j], _mm_shuffle_ps(C4(2, 1), C4(2, 5), 136));
        for (; j < h2; j++) O[j] = I[2 * j] + p*I[2 * j + 1] + I[2 * j + 2];
        if (h % 2 == 0) O[j] = I[2 * j] + (1 + p)*I[2 * j + 1];
    }
    else {
        O[j] = (1 + p)*I[j] + I[j + 1]; j++; if (k == 0) k = (h <= 4) ? h - 1 : 4;
        for (; j < k; j++) O[j] = I[j - 1] + p*I[j] + I[j + 1];
        for (; j < h - 4; j += 4) STR(O[j], C4(1, 0));
        for (; j < h - 1; j++) O[j] = I[j - 1] + p*I[j] + I[j + 1];
        O[j] = I[j - 1] + (1 + p)*I[j];
    }
#undef C4
}

// convolve I by [1 p 1] filter (uses SSE)
void convTri1(float *I, float *O, int h, int w, int d, float p, int s) {
    const float nrm = 1.0f / ((p + 2)*(p + 2)); int i, j, h0 = h - (h % 4);
    float *Il, *Im, *Ir, *T = (float*)alMalloc(h*sizeof(float), 16);
    for (int d0 = 0; d0 < d; d0++) for (i = s / 2; i<w; i += s) {
        Il = Im = Ir = I + i*h + d0*h*w; if (i>0) Il -= h; if (i < w - 1) Ir += h;
        for (j = 0; j < h0; j += 4)
            STR(T[j], MUL(nrm, ADD(ADD(LDu(Il[j]), MUL(p, LDu(Im[j]))), LDu(Ir[j]))));
        for (j = h0; j<h; j++) T[j] = nrm*(Il[j] + p*Im[j] + Ir[j]);
        convTri1Y(T, O, h, p, s); O += h / s;
    }
    alFree(T);
}

// convolve one column of I by a 2rx1 max filter
void convMaxY(float *I, float *O, float *T, int h, int r) {
    int y, y0, y1, yi, m = 2 * r + 1;
#define max1(a,b) a>b ? a : b;
#define maxk(y0,y1) { O[y]=I[y0]; \
    for (yi = y0 + 1; yi <= y1; yi++) { if (I[yi] > O[y]) O[y] = I[yi]; }}
    for (y = 0; y<r; y++) { y1 = y + r; if (y1>h - 1) y1 = h - 1; maxk(0, y1); }
    for (; y <= h - m - r; y += m) {
        T[m - 1] = I[y + r];
        for (yi = 1; yi < m; yi++) T[m - 1 - yi] = max1(T[m - 1 - yi + 1], I[y + r - yi]);
        for (yi = 1; yi < m; yi++) T[m - 1 + yi] = max1(T[m - 1 + yi - 1], I[y + r + yi]);
        for (yi = 0; yi < m; yi++) O[y + yi] = max1(T[yi], T[yi + m - 1]);
    }
    for (; y < h - r; y++) { maxk(y - r, y + r); }
    for (; y<h; y++) { y0 = y - r; if (y0<0) y0 = 0; maxk(y0, h - 1); }
#undef maxk
#undef max1
}

// convolve every column of I by a 2rx1 max filter
void convMax(float *I, float *O, int h, int w, int d, int r) {
    if (r>w - 1) r = w - 1; if (r>h - 1) r = h - 1; int m = 2 * r + 1;
    float *T = (float*)alMalloc(m * 2 * sizeof(float), 16);
    for (int d0 = 0; d0 < d; d0++) for (int x = 0; x < w; x++) {
        float *Oc = O + d0*h*w + h*x, *Ic = I + d0*h*w + h*x;
        convMaxY(Ic, Oc, T, h, r);
    }
    alFree(T);
}

////////////////////////////////////////////////////////////////////
////   gradientMex
/////////////////////////////////////////////////////////////////////

// compute x and y gradients for just one column (uses sse)
void grad1(float *I, float *Gx, float *Gy, int h, int w, int x) {
    int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
    // compute column of Gx
    Ip = I - h; In = I + h; r = .5f;
    if (x == 0) { r = 1; Ip += h; }
    else if (x == w - 1) { r = 1; In -= h; }
    if (h<4 || h % 4>0 || (size_t(I) & 15) || (size_t(Gx) & 15)) {
        for (y = 0; y < h; y++) *Gx++ = (*In++ - *Ip++)*r;
    }
    else {
        _G = (__m128*) Gx; _Ip = (__m128*) Ip; _In = (__m128*) In; _r = SET(r);
        for (y = 0; y<h; y += 4) *_G++ = MUL(SUB(*_In++, *_Ip++), _r);
    }
    // compute column of Gy
#define GRADY(r) *Gy++=(*In++-*Ip++)*r;
    Ip = I; In = Ip + 1;
    // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
    y1 = ((~((size_t)Gy) + 1) & 15) / 4; if (y1 == 0) y1 = 4; if (y1>h - 1) y1 = h - 1;
    GRADY(1); Ip--; for (y = 1; y < y1; y++) GRADY(.5f);
    _r = SET(.5f); _G = (__m128*) Gy;
    for (; y + 4 < h - 1; y += 4, Ip += 4, In += 4, Gy += 4){
        *_G = MUL(SUB(LDu(*In), LDu(*Ip)), _r);
        _G++;
    }
    for (; y < h - 1; y++) GRADY(.5f); In--; GRADY(1);
#undef GRADY
}

// compute x and y gradients at each location (uses sse)
void grad2(float *I, float *Gx, float *Gy, int h, int w, int d) {
    int o, x, c, a = w*h; for (c = 0; c < d; c++) for (x = 0; x < w; x++) {
        o = c*a + x*h; grad1(I + o, Gx + o, Gy + o, h, w, x);
    }
}

// build lookup table a[] s.t. a[x*n]~=acos(x) for x in [-1,1]
float* acosTable() {
    const int n = 10000, b = 10; int i;
    static float a[n * 2 + b * 2]; static bool init = false;
    float *a1 = a + n + b; if (init) return a1;
    for (i = -n - b; i < -n; i++)   a1[i] = PI;
    for (i = -n; i < n; i++)      a1[i] = float(acos(i / float(n)));
    for (i = n; i < n + b; i++)     a1[i] = 0;
    for (i = -n - b; i<n / 10; i++) if (a1[i] > PI - 1e-6f) a1[i] = PI - 1e-6f;
    init = true; return a1;
}

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag(float *I, float *M, float *O, int h, int w, int d, bool full) {
    int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
    float *acost = acosTable(), acMult = 10000.0f;
    // allocate memory for storing one column of output (padded so h4%4==0)
    h4 = (h % 4 == 0) ? h : h - (h % 4) + 4; s = d*h4*sizeof(float);
    M2 = (float*)alMalloc(s, 16); _M2 = (__m128*) M2;
    Gx = (float*)alMalloc(s, 16); _Gx = (__m128*) Gx;
    Gy = (float*)alMalloc(s, 16); _Gy = (__m128*) Gy;
    // compute gradient magnitude and orientation for each column
    for (x = 0; x < w; x++) {
        // compute gradients (Gx, Gy) with maximum squared magnitude (M2)
        for (c = 0; c < d; c++) {
            grad1(I + x*h + c*w*h, Gx + c*h4, Gy + c*h4, h, w, x);
            for (y = 0; y < h4 / 4; y++) {
                y1 = h4 / 4 * c + y;
                _M2[y1] = ADD(MUL(_Gx[y1], _Gx[y1]), MUL(_Gy[y1], _Gy[y1]));
                if (c == 0) continue; _m = CMPGT(_M2[y1], _M2[y]);
                _M2[y] = OR(AND(_m, _M2[y1]), ANDNOT(_m, _M2[y]));
                _Gx[y] = OR(AND(_m, _Gx[y1]), ANDNOT(_m, _Gx[y]));
                _Gy[y] = OR(AND(_m, _Gy[y1]), ANDNOT(_m, _Gy[y]));
            }
        }
        // compute gradient mangitude (M) and normalize Gx
        for (y = 0; y < h4 / 4; y++) {
            _m = MIn(RCPSQRT(_M2[y]), SET(1e10f));
            _M2[y] = RCP(_m);
            if (O) _Gx[y] = MUL(MUL(_Gx[y], _m), SET(acMult));
            if (O) _Gx[y] = XOR(_Gx[y], AND(_Gy[y], SET(-0.f)));
        };
        memcpy(M + x*h, M2, h*sizeof(float));
        // compute and store gradient orientation (O) via table lookup
        if (O != 0) for (y = 0; y < h; y++) O[x*h + y] = acost[(int)Gx[y]];
// 		if (O != 0 && full) {
// 			y1 = ((~size_t(O + x*h) + 1) & 15) / 4; y = 0;
// 			for (; y < y1; y++) O[y + x*h] += (Gy[y] < 0)*PI;
// 			for (; y < h - 4; y += 4) STRu(O[y + x*h],
// 				ADD(LDu(O[y + x*h]), AND(CMPLT(LDu(Gy[y]), SET(0.f)), SET(PI))));
// 			for (; y < h; y++) O[y + x*h] += (Gy[y] < 0)*PI;
// 		}
    }
    alFree(Gx); alFree(Gy); alFree(M2);
}


// normalize gradient magnitude at each location (uses sse)
void gradMagNorm(float *M, float *S, int h, int w, float norm) {
    __m128 *_M, *_S, _norm; int i = 0, n = h*w, n4 = n / 4;
    _S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
    bool sse = !(size_t(M) & 15) && !(size_t(S) & 15);
    if (sse) {
        for (; i < n4; i++){
            *_M = MUL(*_M, RCP(ADD(*_S++, _norm)));
            _M++;
        }
        i *= 4;
    }
    for (; i < n; i++) M[i] /= (S[i] + norm);
}

void gradMagNorm_sse(float *M, float *S, int h, int w, float norm){
    __m128 *_M = (__m128*)M;
    __m128 *_S = (__m128*)S;
    float _norm_ptr[4] = {norm, norm, norm, norm};
    __m128 _norm = _mm_load_ps(_norm_ptr);

    bool sse = !(size_t(M) & 15) && !(size_t(S) & 15);
    int i=0;
    int n =  h*w;
    int n4 = (n>>2);
    if (sse){
        for (; i<n4; i++){
            *_M = MUL(*_M, INV(ADD(*_S++, _norm)));
            _M++;
            //*_M ++ = MUL(*_M, INV(ADD(*_S++, _norm)));
        }
        i = i<<2;
    }
    for (; i < n; i++){
        M[i] = S[i] + norm;
        //M[i] /= (S[i] + norm);
    }
}

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1,
    int nb, int n, float norm, int nOrients, bool full, bool interpolate)
{
    // assumes all *OUTPUT* matrices are 4-byte aligned
    int i, o0, o1; float o, od, m;
    __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;
    // define useful constants
    const float oMult = (float)nOrients / (full ? 2 * PI : PI); const int oMax = nOrients*nb;
    const __m128 _norm = SET(norm), _oMult = SET(oMult), _nbf = SET((float)nb);
    const __m128i _oMax = SET(oMax), _nb = SET(nb);
    // perform the majority of the work with sse
    _O0 = (__m128i*) O0; _O1 = (__m128i*) O1; _M0 = (__m128*) M0; _M1 = (__m128*) M1;
    if (interpolate) for (i = 0; i <= n - 4; i += 4) {
        _o = MUL(LDu(O[i]), _oMult); _o0 = CVT(_o); _od = SUB(_o, CVT(_o0));
        _o0 = CVT(MUL(CVT(_o0), _nbf)); _o0 = AND(CMPGT(_oMax, _o0), _o0); *_O0++ = _o0;
        _o1 = ADD(_o0, _nb); _o1 = AND(CMPGT(_oMax, _o1), _o1); *_O1++ = _o1;
        _m = MUL(LDu(M[i]), _norm); *_M1 = MUL(_od, _m); *_M0++ = SUB(_m, *_M1); _M1++;
    }
    else for (i = 0; i <= n - 4; i += 4) {
        _o = MUL(LDu(O[i]), _oMult); _o0 = CVT(ADD(_o, SET(.5f)));
        _o0 = CVT(MUL(CVT(_o0), _nbf)); _o0 = AND(CMPGT(_oMax, _o0), _o0); *_O0++ = _o0;
        *_M0++ = MUL(LDu(M[i]), _norm); *_M1++ = SET(0.f); *_O1++ = SET(0);
    }
    // compute trailing locations without sse
    if (interpolate) for (i; i < n; i++) {
        o = O[i] * oMult; o0 = (int)o; od = o - o0;
        o0 *= nb; if (o0 >= oMax) o0 = 0; O0[i] = o0;
        o1 = o0 + nb; if (o1 == oMax) o1 = 0; O1[i] = o1;
        m = M[i] * norm; M1[i] = od*m; M0[i] = m - M1[i];
    }
    else for (i; i < n; i++) {
        o = O[i] * oMult; o0 = (int)(o + .5f);
        o0 *= nb; if (o0 >= oMax) o0 = 0; O0[i] = o0;
        M0[i] = M[i] * norm; M1[i] = 0; O1[i] = 0;
    }
}

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist(float *M, float *O, float *H, int h, int w,
    int bin, int nOrients, int softBin, bool full)
{
    const int hb = h / bin, wb = w / bin, h0 = hb*bin, w0 = wb*bin, nb = wb*hb;
    const float s = (float)bin, sInv = 1 / s, sInv2 = 1 / s / s;
    float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1; float xb, init;
    O0 = (int*)alMalloc(h*sizeof(int), 16); M0 = (float*)alMalloc(h*sizeof(float), 16);
    O1 = (int*)alMalloc(h*sizeof(int), 16); M1 = (float*)alMalloc(h*sizeof(float), 16);
    // main loop
    for (x = 0; x < w0; x++) {
        // compute target orientation bins for entire column - very fast
        gradQuantize(O + x*h, M + x*h, O0, O1, M0, M1, nb, h0, sInv2, nOrients, full, softBin >= 0);

        if (softBin < 0 && softBin % 2 == 0) {
            // no interpolation w.r.t. either orienation or spatial bin
            H1 = H + (x / bin)*hb;
#define GH H1[O0[y]]+=M0[y]; y++;
            if (bin == 1)      for (y = 0; y < h0;) { GH; H1++; }
            else if (bin == 2) for (y = 0; y < h0;) { GH; GH; H1++; }
            else if (bin == 3) for (y = 0; y < h0;) { GH; GH; GH; H1++; }
            else if (bin == 4) for (y = 0; y < h0;) { GH; GH; GH; GH; H1++; }
            else for (y = 0; y < h0;) { for (int y1 = 0; y1 < bin; y1++) { GH; } H1++; }
#undef GH

        }
        else if (softBin % 2 == 0 || bin == 1) {
            // interpolate w.r.t. orientation only, not spatial bin
            H1 = H + (x / bin)*hb;
#define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
            if (bin == 1)      for (y = 0; y < h0;) { GH; H1++; }
            else if (bin == 2) for (y = 0; y < h0;) { GH; GH; H1++; }
            else if (bin == 3) for (y = 0; y < h0;) { GH; GH; GH; H1++; }
            else if (bin == 4) for (y = 0; y < h0;) { GH; GH; GH; GH; H1++; }
            else for (y = 0; y < h0;) { for (int y1 = 0; y1 < bin; y1++) { GH; } H1++; }
#undef GH

        }
        else {
            // interpolate using trilinear interpolation
            float ms[4], xyd, yb, xd, yd; __m128 _m, _m0, _m1;
            bool hasLf, hasRt; int xb0, yb0;
            if (x == 0) { init = (0 + .5f)*sInv - 0.5f; xb = init; }
            hasLf = xb >= 0; xb0 = hasLf ? (int)xb : -1; hasRt = xb0 < wb - 1;
            xd = xb - xb0; xb += sInv; yb = init; y = 0;
            // macros for code conciseness
#define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
    ms[0] = 1 - xd - yd + xyd; ms[1] = yd - xyd; ms[2] = xd - xyd; ms[3] = xyd;
#define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));
            // leading rows, no top bin
            for (; y < bin / 2; y++) {
                yb0 = -1; GHinit;
                if (hasLf) { H0[O0[y] + 1] += ms[1] * M0[y]; H0[O1[y] + 1] += ms[1] * M1[y]; }
                if (hasRt) { H0[O0[y] + hb + 1] += ms[3] * M0[y]; H0[O1[y] + hb + 1] += ms[3] * M1[y]; }
            }
            // main rows, has top and bottom bins, use SSE for minor speedup
            if (softBin < 0) for (;; y++) {
                yb0 = (int)yb; if (yb0 >= hb - 1) break; GHinit; _m0 = SET(M0[y]);
                if (hasLf) { _m = SET(0, 0, ms[1], ms[0]); GH(H0 + O0[y], _m, _m0); }
                if (hasRt) { _m = SET(0, 0, ms[3], ms[2]); GH(H0 + O0[y] + hb, _m, _m0); }
            }
            else for (;; y++) {
                yb0 = (int)yb; if (yb0 >= hb - 1) break; GHinit;
                _m0 = SET(M0[y]); _m1 = SET(M1[y]);
                if (hasLf) {
                    _m = SET(0, 0, ms[1], ms[0]);
                    GH(H0 + O0[y], _m, _m0); GH(H0 + O1[y], _m, _m1);
                }
                if (hasRt) {
                    _m = SET(0, 0, ms[3], ms[2]);
                    GH(H0 + O0[y] + hb, _m, _m0); GH(H0 + O1[y] + hb, _m, _m1);
                }
            }
            // final rows, no bottom bin
            for (; y < h0; y++) {
                yb0 = (int)yb; GHinit;
                if (hasLf) { H0[O0[y]] += ms[0] * M0[y]; H0[O1[y]] += ms[0] * M1[y]; }
                if (hasRt) { H0[O0[y] + hb] += ms[2] * M0[y]; H0[O1[y] + hb] += ms[2] * M1[y]; }
            }
#undef GHinit
#undef GH
        }
    }
    alFree(O0); alFree(O1); alFree(M0); alFree(M1);
    // normalize boundary bins which only get 7/8 of weight of interior bins
    if (softBin % 2 != 0) for (int o = 0; o < nOrients; o++) {
        x = 0; for (y = 0; y < hb; y++) H[o*nb + x*hb + y] *= 8.f / 7.f;
        y = 0; for (x = 0; x < wb; x++) H[o*nb + x*hb + y] *= 8.f / 7.f;
        x = wb - 1; for (y = 0; y < hb; y++) H[o*nb + x*hb + y] *= 8.f / 7.f;
        y = hb - 1; for (x = 0; x < wb; x++) H[o*nb + x*hb + y] *= 8.f / 7.f;
    }
}

/******************************************************************************/

// HOG helper: compute 2x2 block normalization values (padded by 1 pixel)
float* hogNormMatrix(float *H, int nOrients, int hb, int wb, int bin) {
    float *N, *N1, *n; int o, x, y, dx, dy, hb1 = hb + 1, wb1 = wb + 1;
    float eps = 1e-4f / 4 / bin / bin / bin / bin; // precise backward equality
    N = (float*)wrCalloc(hb1*wb1, sizeof(float)); N1 = N + hb1 + 1;
    for (o = 0; o < nOrients; o++) for (x = 0; x < wb; x++) for (y = 0; y < hb; y++)
        N1[x*hb1 + y] += H[o*wb*hb + x*hb + y] * H[o*wb*hb + x*hb + y];
    for (x = 0; x < wb - 1; x++) for (y = 0; y < hb - 1; y++) {
        n = N1 + x*hb1 + y; *n = 1 / float(sqrt(n[0] + n[1] + n[hb1] + n[hb1 + 1] + eps));
    }
    x = 0;     dx = 1; dy = 1; y = 0;                  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = 0;     dx = 1; dy = 0; for (y = 0; y < hb1; y++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = 0;     dx = 1; dy = -1; y = hb1 - 1;              N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = wb1 - 1; dx = -1; dy = 1; y = 0;                  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = wb1 - 1; dx = -1; dy = 0; for (y = 0; y < hb1; y++) N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    x = wb1 - 1; dx = -1; dy = -1; y = hb1 - 1;              N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    y = 0;     dx = 0; dy = 1; for (x = 0; x < wb1; x++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    y = hb1 - 1; dx = 0; dy = -1; for (x = 0; x<wb1; x++)  N[x*hb1 + y] = N[(x + dx)*hb1 + y + dy];
    return N;
}

// HOG helper: compute HOG or FHOG channels
void hogChannels(float *H, const float *R, const float *N,
    int hb, int wb, int nOrients, float clip, int type)
{
#define GETT(blk) t=R1[y]*N1[y-(blk)]; if(t>clip) t=clip; c++;
    const float r = .2357f; int o, x, y, c; float t;
    const int nb = wb*hb, nbo = nOrients*nb, hb1 = hb + 1;
    for (o = 0; o < nOrients; o++) for (x = 0; x < wb; x++) {
        const float *R1 = R + o*nb + x*hb, *N1 = N + x*hb1 + hb1 + 1;
        float *H1 = (type <= 1) ? (H + o*nb + x*hb) : (H + x*hb);
        if (type == 0) for (y = 0; y < hb; y++) {
            // store each orientation and normalization (nOrients*4 channels)
            c = -1; GETT(0); H1[c*nbo + y] = t; GETT(1); H1[c*nbo + y] = t;
            GETT(hb1); H1[c*nbo + y] = t; GETT(hb1 + 1); H1[c*nbo + y] = t;
        }
        else if (type == 1) for (y = 0; y < hb; y++) {
            // sum across all normalizations (nOrients channels)
            c = -1; GETT(0); H1[y] += t*.5f; GETT(1); H1[y] += t*.5f;
            GETT(hb1); H1[y] += t*.5f; GETT(hb1 + 1); H1[y] += t*.5f;
        }
        else if (type == 2) for (y = 0; y < hb; y++) {
            // sum across all orientations (4 channels)
            c = -1; GETT(0); H1[c*nb + y] += t*r; GETT(1); H1[c*nb + y] += t*r;
            GETT(hb1); H1[c*nb + y] += t*r; GETT(hb1 + 1); H1[c*nb + y] += t*r;
        }
    }
#undef GETT
}

// compute HOG features
void hog(float *M, float *O, float *H, int h, int w, int binSize,
    int nOrients, int softBin, bool full, float clip)
{
    float *N, *R; const int hb = h / binSize, wb = w / binSize, nb = hb*wb;
    // compute unnormalized gradient histograms
    R = (float*)wrCalloc(wb*hb*nOrients, sizeof(float));
    gradHist(M, O, R, h, w, binSize, nOrients, softBin, full);
    // compute block normalization values
    N = hogNormMatrix(R, nOrients, hb, wb, binSize);
    // perform four normalizations per spatial block
    hogChannels(H, R, N, hb, wb, nOrients, clip, 0);
    wrFree(N); wrFree(R);
}

// compute FHOG features
void fhog(float *M, float *O, float *H, int h, int w, int binSize,
    int nOrients, int softBin, float clip)
{
    const int hb = h / binSize, wb = w / binSize, nb = hb*wb, nbo = nb*nOrients;
    float *N, *R1, *R2; int o, x;
    // compute unnormalized constrast sensitive histograms
    R1 = (float*)wrCalloc(wb*hb*nOrients * 2, sizeof(float));
    gradHist(M, O, R1, h, w, binSize, nOrients * 2, softBin, true);
    // compute unnormalized contrast insensitive histograms
    R2 = (float*)wrCalloc(wb*hb*nOrients, sizeof(float));
    for (o = 0; o < nOrients; o++) for (x = 0; x < nb; x++)
        R2[o*nb + x] = R1[o*nb + x] + R1[(o + nOrients)*nb + x];
    // compute block normalization values
    N = hogNormMatrix(R2, nOrients, hb, wb, binSize);
    // normalized histograms and texture channels
    hogChannels(H + nbo * 0, R1, N, hb, wb, nOrients * 2, clip, 1);
    hogChannels(H + nbo * 2, R2, N, hb, wb, nOrients * 1, clip, 1);
    hogChannels(H + nbo * 3, R1, N, hb, wb, nOrients * 2, clip, 2);
    wrFree(N); wrFree(R1); wrFree(R2);
}

/*
// compute x and y gradients for just one column (uses sse)
void grad1(float *I, float *Gx, float *Gy, int h, int w, int x) {
    int y, y1; float *Ip, *In, r; __m128 *_Ip, *_In, *_G, _r;
    // compute column of Gx
    Ip = I - h; In = I + h; r = .5f;
    if (x == 0) { r = 1; Ip += h; }
    else if (x == w - 1) { r = 1; In -= h; }
    if (h<4 || h % 4>0 || (size_t(I) & 15) || (size_t(Gx) & 15)) {
        for (y = 0; y < h; y++) *Gx++ = (*In++ - *Ip++)*r;
    }
    else {
        _G = (__m128*) Gx; _Ip = (__m128*) Ip; _In = (__m128*) In; _r = SET(r);
        for (y = 0; y<h; y += 4) *_G++ = MUL(SUB(*_In++, *_Ip++), _r);
    }
    // compute column of Gy
#define GRADY(r) *Gy++=(*In++-*Ip++)*r;
    Ip = I; In = Ip + 1;
    // GRADY(1); Ip--; for(y=1; y<h-1; y++) GRADY(.5f); In--; GRADY(1);
    y1 = ((~((size_t)Gy) + 1) & 15) / 4; if (y1 == 0) y1 = 4; if (y1>h - 1) y1 = h - 1;
    GRADY(1); Ip--; for (y = 1; y < y1; y++) GRADY(.5f);
    _r = SET(.5f); _G = (__m128*) Gy;
    for (; y + 4 < h - 1; y += 4, Ip += 4, In += 4, Gy += 4)
        *_G++ = MUL(SUB(LDu(*In), LDu(*Ip)), _r);
    for (; y < h - 1; y++) GRADY(.5f); In--; GRADY(1);
#undef GRADY
}

// compute x and y gradients at each location (uses sse)
void grad2(float *I, float *Gx, float *Gy, int h, int w, int d) {
    int o, x, c, a = w*h; for (c = 0; c < d; c++) for (x = 0; x < w; x++) {
        o = c*a + x*h; grad1(I + o, Gx + o, Gy + o, h, w, x);
    }
}

// build lookup table a[] s.t. a[dx/2.02*n]~=acos(dx)
float* acosTable() {
    int i, n = 25000, n2 = n / 2; float t, ni;
    static float a[25000]; static bool init = false;
    if (init) return a + n2; ni = 2.02f / (float)n;
    for (i = 0; i < n; i++) {
        t = i*ni - 1.01f;
        t = t<-1 ? -1 : (t>1 ? 1 : t);
        t = (float)acos(t);
        a[i] = (t <= PI - 1e-5f) ? t : 0;
    }
    init = true; return a + n2;
}

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag(float *I, float *M, float *O, int h, int w, int d) {
    int x, y, y1, c, h4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
    float *acost = acosTable(), acMult = 25000 / 2.02f;
    // allocate memory for storing one column of output (padded so h4%4==0)
    h4 = (h % 4 == 0) ? h : h - (h % 4) + 4; s = d*h4*sizeof(float);
    M2 = (float*)alMalloc(s, 16); _M2 = (__m128*) M2;
    Gx = (float*)alMalloc(s, 16); _Gx = (__m128*) Gx;
    Gy = (float*)alMalloc(s, 16); _Gy = (__m128*) Gy;
    // compute gradient magnitude and orientation for each column
    for (x = 0; x < w; x++) {
        // compute gradients (Gx, Gy) and squared magnitude (M2) for each channel
        for (c = 0; c < d; c++) grad1(I + x*h + c*w*h, Gx + c*h4, Gy + c*h4, h, w, x);
        for (y = 0; y < d*h4 / 4; y++) _M2[y] = ADD(MUL(_Gx[y], _Gx[y]), MUL(_Gy[y], _Gy[y]));
        // store gradients with maximum response in the first channel
        for (c = 1; c < d; c++) {
            for (y = 0; y < h4 / 4; y++) {
                y1 = h4 / 4 * c + y; _m = CMPGT(_M2[y1], _M2[y]);
                _M2[y] = OR(AND(_m, _M2[y1]), ANDNOT(_m, _M2[y]));
                _Gx[y] = OR(AND(_m, _Gx[y1]), ANDNOT(_m, _Gx[y]));
                _Gy[y] = OR(AND(_m, _Gy[y1]), ANDNOT(_m, _Gy[y]));
            }
        }
        // compute gradient mangitude (M) and normalize Gx
        for (y = 0; y < h4 / 4; y++) {
            _m = MIN(RCPSQRT(_M2[y]), SET(1e10f));
            _M2[y] = RCP(_m);
            _Gx[y] = MUL(MUL(_Gx[y], _m), SET(acMult));
            _Gx[y] = XOR(_Gx[y], AND(_Gy[y], SET(-0.f)));
        };
        memcpy(M + x*h, M2, h*sizeof(float));
        // compute and store gradient orientation (O) via table lookup
        if (O != 0) for (y = 0; y < h; y++) O[x*h + y] = acost[(int)Gx[y]];
    }
    alFree(Gx); alFree(Gy); alFree(M2);
}

// normalize gradient magnitude at each location (uses sse)
void gradMagNorm(float *M, float *S, int h, int w, float norm) {
    __m128 *_M, *_S, _norm; int i = 0, n = h*w, n4 = n / 4;
    _S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
    bool sse = !(size_t(M) & 15) && !(size_t(S) & 15);
    if (sse) { for (; i < n4; i++) *_M++ = MUL(*_M, RCP(ADD(*_S++, _norm))); i *= 4; }
    for (; i < n; i++) M[i] /= (S[i] + norm);
}

// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize(float *O, float *M, int *O0, int *O1, float *M0, float *M1,
    int nOrients, int nb, int n, float norm)
{
    // assumes all *OUTPUT* matrices are 4-byte aligned
    int i, o0, o1; float o, od, m;
    __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _o0f, _m, *_M0, *_M1;
    // define useful constants
    const float oMult = (float)nOrients / PI; const int oMax = nOrients*nb;
    const __m128 _norm = SET(norm), _oMult = SET(oMult), _nbf = SET((float)nb);
    const __m128i _oMax = SET(oMax), _nb = SET(nb);
    // perform the majority of the work with sse
    _O0 = (__m128i*) O0; _O1 = (__m128i*) O1; _M0 = (__m128*) M0; _M1 = (__m128*) M1;
    for (i = 0; i <= n - 4; i += 4) {
        _o = MUL(LDu(O[i]), _oMult); _o0f = CVT(CVT(_o)); _o0 = CVT(MUL(_o0f, _nbf));
        _o1 = ADD(_o0, _nb); _o1 = AND(CMPGT(_oMax, _o1), _o1);
        *_O0++ = _o0; *_O1++ = _o1; _m = MUL(LDu(M[i]), _norm);
        *_M1 = MUL(SUB(_o, _o0f), _m); *_M0 = SUB(_m, *_M1); _M0++; _M1++;
    }
    // compute trailing locations without sse
    for (i; i < n; i++) {
        o = O[i] * oMult; m = M[i] * norm; o0 = (int)o; od = o - o0;
        o0 *= nb; o1 = o0 + nb; if (o1 == oMax) o1 = 0;
        O0[i] = o0; O1[i] = o1; M1[i] = od*m; M0[i] = m - M1[i];
    }
}

// compute nOrients gradient histograms per bin x bin block of pixels
void gradHist(float *M, float *O, float *H, int h, int w,
    int bin, int nOrients, bool softBin)
{
    const int hb = h / bin, wb = w / bin, h0 = hb*bin, w0 = wb*bin, nb = wb*hb;
    const float s = (float)bin, sInv = 1 / s, sInv2 = 1 / s / s;
    float *H0, *H1, *M0, *M1; int x, y; int *O0, *O1;
    O0 = (int*)alMalloc(h*sizeof(int), 16); M0 = (float*)alMalloc(h*sizeof(float), 16);
    O1 = (int*)alMalloc(h*sizeof(int), 16); M1 = (float*)alMalloc(h*sizeof(float), 16);
    // main loop
    for (x = 0; x < w0; x++) {
        // compute target orientation bins for entire column - very fast
        gradQuantize(O + x*h, M + x*h, O0, O1, M0, M1, nOrients, nb, h0, sInv2);

        if (!softBin || bin == 1) {
            // interpolate w.r.t. orientation only, not spatial bin
            H1 = H + (x / bin)*hb;
#define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
            if (bin == 1)      for (y = 0; y < h0;) { GH; H1++; }
            else if (bin == 2) for (y = 0; y < h0;) { GH; GH; H1++; }
            else if (bin == 3) for (y = 0; y < h0;) { GH; GH; GH; H1++; }
            else if (bin == 4) for (y = 0; y < h0;) { GH; GH; GH; GH; H1++; }
            else for (y = 0; y < h0;) { for (int y1 = 0; y1 < bin; y1++) { GH; } H1++; }
#undef GH

        }
        else {
            // interpolate using trilinear interpolation
            float ms[4], xyd, xb, yb, xd, yd, init; __m128 _m, _m0, _m1;
            bool hasLf, hasRt; int xb0, yb0;
            if (x == 0) { init = (0 + .5f)*sInv - 0.5f; xb = init; }
            hasLf = xb >= 0; xb0 = hasLf ? (int)xb : -1; hasRt = xb0 < wb - 1;
            xd = xb - xb0; xb += sInv; yb = init; y = 0;
            // macros for code conciseness
#define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
    ms[0] = 1 - xd - yd + xyd; ms[1] = yd - xyd; ms[2] = xd - xyd; ms[3] = xyd;
#define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));
            // leading rows, no top bin
            for (; y < bin / 2; y++) {
                yb0 = -1; GHinit;
                if (hasLf) { H0[O0[y] + 1] += ms[1] * M0[y]; H0[O1[y] + 1] += ms[1] * M1[y]; }
                if (hasRt) { H0[O0[y] + hb + 1] += ms[3] * M0[y]; H0[O1[y] + hb + 1] += ms[3] * M1[y]; }
            }
            // main rows, has top and bottom bins, use SSE for minor speedup
            for (;; y++) {
                yb0 = (int)yb; if (yb0 >= hb - 1) break; GHinit;
                _m0 = SET(M0[y]); _m1 = SET(M1[y]);
                if (hasLf) {
                    _m = SET(0, 0, ms[1], ms[0]);
                    GH(H0 + O0[y], _m, _m0); GH(H0 + O1[y], _m, _m1);
                }
                if (hasRt) {
                    _m = SET(0, 0, ms[3], ms[2]);
                    GH(H0 + O0[y] + hb, _m, _m0); GH(H0 + O1[y] + hb, _m, _m1);
                }
            }
            // final rows, no bottom bin
            for (; y < h0; y++) {
                yb0 = (int)yb; GHinit;
                if (hasLf) { H0[O0[y]] += ms[0] * M0[y]; H0[O1[y]] += ms[0] * M1[y]; }
                if (hasRt) { H0[O0[y] + hb] += ms[2] * M0[y]; H0[O1[y] + hb] += ms[2] * M1[y]; }
            }
#undef GHinit
#undef GH
        }
    }
    alFree(O0); alFree(O1); alFree(M0); alFree(M1);
}

// compute HOG features given gradient histograms
void hog(float *H, float *G, int h, int w, int bin, int nOrients, float clip){
    float *N, *N1, *H1; int o, x, y, hb = h / bin, wb = w / bin, nb = wb*hb;
    float eps = 1e-4f / 4 / bin / bin / bin / bin; // precise backward equality
    // compute 2x2 block normalization values
    N = (float*)wrCalloc(nb, sizeof(float));
    for (o = 0; o < nOrients; o++) for (x = 0; x < nb; x++) N[x] += H[x + o*nb] * H[x + o*nb];
    for (x = 0; x < wb - 1; x++) for (y = 0; y<hb - 1; y++) {
        N1 = N + x*hb + y; *N1 = 1 / float(sqrt(N1[0] + N1[1] + N1[hb] + N1[hb + 1] + eps));
    }
    // perform 4 normalizations per spatial block (handling boundary regions)
#define U(a,b) Gs[a][y]=H1[y]*N1[y-(b)]; if(Gs[a][y]>clip) Gs[a][y]=clip;
    for (o = 0; o < nOrients; o++) for (x = 0; x < wb; x++) {
        H1 = H + o*nb + x*hb; N1 = N + x*hb; float *Gs[4]; Gs[0] = G + o*nb + x*hb;
        for (y = 1; y < 4; y++) Gs[y] = Gs[y - 1] + nb*nOrients;
        bool lf, md, rt; lf = (x == 0); rt = (x == wb - 1); md = (!lf && !rt);
        y = 0; if (!rt) U(0, 0); if (!lf) U(2, hb);
        if (lf) for (y = 1; y < hb - 1; y++) { U(0, 0); U(1, 1); }
        if (md) for (y = 1; y < hb - 1; y++) { U(0, 0); U(1, 1); U(2, hb); U(3, hb + 1); }
        if (rt) for (y = 1; y < hb - 1; y++) { U(2, hb); U(3, hb + 1); }
        y = hb - 1; if (!rt) U(1, 1); if (!lf) U(3, hb + 1);
    } wrFree(N);
#undef U
}
*/

/*
////////////////////////////////////////////////////////////////////
////   imPadMex
/////////////////////////////////////////////////////////////////////
typedef unsigned char uchar;

// pad A by [pt,pb,pl,pr] and store result in B
template<class T> void imPad(T *A, T *B, int h, int w, int d, int pt, int pb,
    int pl, int pr, int flag, T val)
{
    int h1 = h + pt, hb = h1 + pb, w1 = w + pl, wb = w1 + pr, x, y, z, mPad;
    int ct = 0, cb = 0, cl = 0, cr = 0;
    if (pt<0) { ct = -pt; pt = 0; } if (pb<0) { h1 += pb; cb = -pb; pb = 0; }
    if (pl<0) { cl = -pl; pl = 0; } if (pr<0) { w1 += pr; cr = -pr; pr = 0; }
    int *xs, *ys; x = pr>pl ? pr : pl; y = pt>pb ? pt : pb; mPad = x>y ? x : y;
    bool useLookup = ((flag == 2 || flag == 3) && (mPad>h || mPad > w))
        || (flag == 3 && (ct || cb || cl || cr));
    // helper macro for padding
#define PAD(XL,XM,XR,YT,YM,YB) \
    for (x = 0; x < pl; x++) for (y = 0; y < pt; y++) B[x*hb + y] = A[(XL + cl)*h + YT + ct]; \
    for (x = 0; x < pl; x++) for (y = pt; y < h1; y++) B[x*hb + y] = A[(XL + cl)*h + YM + ct]; \
    for (x = 0; x < pl; x++) for (y = h1; y < hb; y++) B[x*hb + y] = A[(XL + cl)*h + YB - cb]; \
    for (x = pl; x < w1; x++) for (y = 0; y < pt; y++) B[x*hb + y] = A[(XM + cl)*h + YT + ct]; \
    for (x = pl; x < w1; x++) for (y = h1; y < hb; y++) B[x*hb + y] = A[(XM + cl)*h + YB - cb]; \
    for (x = w1; x < wb; x++) for (y = 0; y < pt; y++) B[x*hb + y] = A[(XR - cr)*h + YT + ct]; \
    for (x = w1; x < wb; x++) for (y = pt; y < h1; y++) B[x*hb + y] = A[(XR - cr)*h + YM + ct]; \
    for (x = w1; x < wb; x++) for (y = h1; y < hb; y++) B[x*hb + y] = A[(XR - cr)*h + YB - cb];
    // build lookup table for xs and ys if necessary
    if (useLookup) {
        xs = (int*)wrMalloc(wb*sizeof(int)); int h2 = (pt + 1) * 2 * h;
        ys = (int*)wrMalloc(hb*sizeof(int)); int w2 = (pl + 1) * 2 * w;
        if (flag == 2) {
            for (x = 0; x < wb; x++) { z = (x - pl + w2) % (w * 2); xs[x] = z < w ? z : w * 2 - z - 1; }
            for (y = 0; y < hb; y++) { z = (y - pt + h2) % (h * 2); ys[y] = z < h ? z : h * 2 - z - 1; }
        }
        else if (flag == 3) {
            for (x = 0; x < wb; x++) xs[x] = (x - pl + w2) % w;
            for (y = 0; y < hb; y++) ys[y] = (y - pt + h2) % h;
        }
    }
    // pad by appropriate value
    for (z = 0; z < d; z++) {
        // copy over A to relevant region in B
        for (x = 0; x < w - cr - cl; x++)
            memcpy(B + (x + pl)*hb + pt, A + (x + cl)*h + ct, sizeof(T)*(h - ct - cb));
        // set boundaries of B to appropriate values
        if (flag == 0 && val != 0) { // "constant"
            for (x = 0; x < pl; x++) for (y = 0; y < hb; y++) B[x*hb + y] = val;
            for (x = pl; x < w1; x++) for (y = 0; y < pt; y++) B[x*hb + y] = val;
            for (x = pl; x < w1; x++) for (y = h1; y < hb; y++) B[x*hb + y] = val;
            for (x = w1; x < wb; x++) for (y = 0; y < hb; y++) B[x*hb + y] = val;
        }
        else if (useLookup) { // "lookup"
            PAD(xs[x], xs[x], xs[x], ys[y], ys[y], ys[y]);
        }
        else if (flag == 1) {  // "replicate"
            PAD(0, x - pl, w - 1, 0, y - pt, h - 1);
        }
        else if (flag == 2) { // "symmetric"
            PAD(pl - x - 1, x - pl, w + w1 - 1 - x, pt - y - 1, y - pt, h + h1 - 1 - y);
        }
        else if (flag == 3) { // "circular"
            PAD(x - pl + w, x - pl, x - pl - w, y - pt + h, y - pt, y - pt - h);
        }
        A += h*w;  B += hb*wb;
    }
    if (useLookup) { wrFree(xs); wrFree(ys); }
#undef PAD
}

////////////////////////////////////////////////////////////////////
////   imResampleMex
/////////////////////////////////////////////////////////////////////

// compute interpolation values for single column for resapling
template<class T> void resampleCoef(int ha, int hb, int &n, int *&yas,
    int *&ybs, T *&wts, int bd[2], int pad)
{
    const T s = T(hb) / T(ha), sInv = 1 / s; T wt, wt0 = T(1e-3)*s;
    bool ds = ha > hb; int nMax; bd[0] = bd[1] = 0;
    if (ds) { n = 0; nMax = ha + (pad > 2 ? pad : 2)*hb; }
    else { n = nMax = hb; }
    // initialize memory
    wts = (T*)alMalloc(nMax*sizeof(T), 16);
    yas = (int*)alMalloc(nMax*sizeof(int), 16);
    ybs = (int*)alMalloc(nMax*sizeof(int), 16);
    if (ds) for (int yb = 0; yb<hb; yb++) {
        // create coefficients for downsampling
        T ya0f = yb*sInv, ya1f = ya0f + sInv, W = 0;
        int ya0 = int(ceil(ya0f)), ya1 = int(ya1f), n1 = 0;
        for (int ya = ya0 - 1; ya<ya1 + 1; ya++) {
            wt = s; if (ya == ya0 - 1) wt = (ya0 - ya0f)*s; else if (ya == ya1) wt = (ya1f - ya1)*s;
            if (wt>wt0 && ya >= 0) { ybs[n] = yb; yas[n] = ya; wts[n] = wt; n++; n1++; W += wt; }
        }
        if (W>1) for (int i = 0; i<n1; i++) wts[n - n1 + i] /= W;
        if (n1>bd[0]) bd[0] = n1;
        while (n1 < pad) { ybs[n] = yb; yas[n] = yas[n - 1]; wts[n] = 0; n++; n1++; }
    }
    else for (int yb = 0; yb < hb; yb++) {
        // create coefficients for upsampling
        T yaf = (T(.5) + yb)*sInv - T(.5); int ya = (int)floor(yaf);
        wt = 1; if (ya >= 0 && ya < ha - 1) wt = 1 - (yaf - ya);
        if (ya < 0) { ya = 0; bd[0]++; } if (ya >= ha - 1) { ya = ha - 1; bd[1]++; }
        ybs[yb] = yb; yas[yb] = ya; wts[yb] = wt;
    }
}

// resample A using bilinear interpolation and and store result in B
template<class T>
void resample(T *A, T *B, int ha, int hb, int wa, int wb, int d, T r) {
    int hn, wn, x, x1, y, z, xa, xb, ya; T *A0, *A1, *A2, *A3, *B0, wt, wt1;
    T *C = (T*)alMalloc((ha + 4)*sizeof(T), 16); for (y = ha; y < ha + 4; y++) C[y] = 0;
    bool sse = (typeid(T) == typeid(float)) && !(size_t(A) & 15) && !(size_t(B) & 15);
    // get coefficients for resampling along w and h
    int *xas, *xbs, *yas, *ybs; T *xwts, *ywts; int xbd[2], ybd[2];
    resampleCoef<T>(wa, wb, wn, xas, xbs, xwts, xbd, 0);
    resampleCoef<T>(ha, hb, hn, yas, ybs, ywts, ybd, 4);
    if (wa == 2 * wb) r /= 2; if (wa == 3 * wb) r /= 3; if (wa == 4 * wb) r /= 4;
    r /= T(1 + 1e-6); for (y = 0; y < hn; y++) ywts[y] *= r;
    // resample each channel in turn
    for (z = 0; z < d; z++) for (x = 0; x < wb; x++) {
        if (x == 0) x1 = 0; xa = xas[x1]; xb = xbs[x1]; wt = xwts[x1]; wt1 = 1 - wt; y = 0;
        A0 = A + z*ha*wa + xa*ha; A1 = A0 + ha, A2 = A1 + ha, A3 = A2 + ha; B0 = B + z*hb*wb + xb*hb;
        // variables for SSE (simple casts to float)
        float *Af0, *Af1, *Af2, *Af3, *Bf0, *Cf, *ywtsf, wtf, wt1f;
        Af0 = (float*)A0; Af1 = (float*)A1; Af2 = (float*)A2; Af3 = (float*)A3;
        Bf0 = (float*)B0; Cf = (float*)C;
        ywtsf = (float*)ywts; wtf = (float)wt; wt1f = (float)wt1;
        // resample along x direction (A -> C)
#define FORs(X) if(sse) for(; y<ha-4; y+=4) STR(Cf[y],X);
#define FORr(X) for(; y<ha; y++) C[y] = X;
        if (wa == 2 * wb) {
            FORs(ADD(LDu(Af0[y]), LDu(Af1[y])));
            FORr(A0[y] + A1[y]); x1 += 2;
        }
        else if (wa == 3 * wb) {
            FORs(ADD(LDu(Af0[y]), LDu(Af1[y]), LDu(Af2[y])));
            FORr(A0[y] + A1[y] + A2[y]); x1 += 3;
        }
        else if (wa == 4 * wb) {
            FORs(ADD(LDu(Af0[y]), LDu(Af1[y]), LDu(Af2[y]), LDu(Af3[y])));
            FORr(A0[y] + A1[y] + A2[y] + A3[y]); x1 += 4;
        }
        else if (wa > wb) {
            int m = 1; while (x1 + m < wn && xb == xbs[x1 + m]) m++; float wtsf[4];
            for (int x0 = 0; x0 < (m < 4 ? m : 4); x0++) wtsf[x0] = float(xwts[x1 + x0]);
#define U(x) MUL( LDu(*(Af ## x + y)), SET(wtsf[x]) )
#define V(x) *(A ## x + y) * xwts[x1+x]
            if (m == 1) { FORs(U(0));                     FORr(V(0)); }
            if (m == 2) { FORs(ADD(U(0), U(1)));           FORr(V(0) + V(1)); }
            if (m == 3) { FORs(ADD(U(0), U(1), U(2)));      FORr(V(0) + V(1) + V(2)); }
            if (m >= 4) { FORs(ADD(U(0), U(1), U(2), U(3))); FORr(V(0) + V(1) + V(2) + V(3)); }
#undef U
#undef V
            for (int x0 = 4; x0 < m; x0++) {
                A1 = A0 + x0*ha; wt1 = xwts[x1 + x0]; Af1 = (float*)A1; wt1f = float(wt1); y = 0;
                FORs(ADD(LD(Cf[y]), MUL(LDu(Af1[y]), SET(wt1f)))); FORr(C[y] + A1[y] * wt1);
            }
            x1 += m;
        }
        else {
            bool xBd = x < xbd[0] || x >= wb - xbd[1]; x1++;
            if (xBd) memcpy(C, A0, ha*sizeof(T));
            if (!xBd) FORs(ADD(MUL(LDu(Af0[y]), SET(wtf)), MUL(LDu(Af1[y]), SET(wt1f))));
            if (!xBd) FORr(A0[y] * wt + A1[y] * wt1);
        }
#undef FORs
#undef FORr
        // resample along y direction (B -> C)
        if (ha == hb * 2) {
            T r2 = r / 2; int k = ((~((size_t)B0) + 1) & 15) / 4; y = 0;
            for (; y < k; y++)  B0[y] = (C[2 * y] + C[2 * y + 1])*r2;
            if (sse) for (; y < hb - 4; y += 4) STR(Bf0[y], MUL((float)r2, _mm_shuffle_ps(ADD(
                LDu(Cf[2 * y]), LDu(Cf[2 * y + 1])), ADD(LDu(Cf[2 * y + 4]), LDu(Cf[2 * y + 5])), 136)));
            for (; y < hb; y++) B0[y] = (C[2 * y] + C[2 * y + 1])*r2;
        }
        else if (ha == hb * 3) {
            for (y = 0; y < hb; y++) B0[y] = (C[3 * y] + C[3 * y + 1] + C[3 * y + 2])*(r / 3);
        }
        else if (ha == hb * 4) {
            for (y = 0; y<hb; y++) B0[y] = (C[4 * y] + C[4 * y + 1] + C[4 * y + 2] + C[4 * y + 3])*(r / 4);
        }
        else if (ha>hb) {
            y = 0;
            //if( sse && ybd[0]<=4 ) for(; y<hb; y++) // Requires SSE4
            //  STR1(Bf0[y],_mm_dp_ps(LDu(Cf[yas[y*4]]),LDu(ywtsf[y*4]),0xF1));
#define U(o) C[ya+o]*ywts[y*4+o]
            if (ybd[0] == 2) for (; y < hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1); }
            if (ybd[0] == 3) for (; y < hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1) + U(2); }
            if (ybd[0] == 4) for (; y<hb; y++) { ya = yas[y * 4]; B0[y] = U(0) + U(1) + U(2) + U(3); }
            if (ybd[0]>4)  for (; y < hn; y++) { B0[ybs[y]] += C[yas[y]] * ywts[y]; }
#undef U
        }
        else {
            for (y = 0; y < ybd[0]; y++) B0[y] = C[yas[y]] * ywts[y];
            for (; y < hb - ybd[1]; y++) B0[y] = C[yas[y]] * ywts[y] + C[yas[y] + 1] * (r - ywts[y]);
            for (; y < hb; y++)        B0[y] = C[yas[y]] * ywts[y];
        }
    }
    alFree(xas); alFree(xbs); alFree(xwts); alFree(C);
    alFree(yas); alFree(ybs); alFree(ywts);
}
*/
