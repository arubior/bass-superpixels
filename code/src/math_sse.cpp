#include "math_sse.hpp"
#include <xmmintrin.h>
#include <emmintrin.h>
#include <cmath>

namespace MathSSE{

inline __m128 abs_ps(__m128 x) {
    static const __m128 sign_mask = _mm_set1_ps(-0.f); // -0.f = 1 << 31
    return _mm_andnot_ps(sign_mask, x);
}

inline __m128d abs_pd(__m128d x) {
    static const __m128d sign_mask = _mm_set1_pd(-0.); // -0. = 1 << 63
    return _mm_andnot_pd(sign_mask, x); // !sign_mask & x
}



float norm_l1(const float *p, const int n){

    float norm;
    float r[4];
    __m128 *ptr_p = (__m128*)p;
    __m128 acc = {0.f, 0.f, 0.f, 0.f};
    int i=0;
    for (; i<n/4; ++i){
        acc = _mm_add_ps(acc, abs_ps(ptr_p[i]) );
    }
    _mm_store_ps(r, acc);
    norm = r[0] + r[1] + r[2] + r[3];
    if ( n&3){
        for (i=n-(n&3); i<n; ++i)
            norm += std::fabs(p[i]);
    }
    return norm;
}


void normalize_l1(const float *p, float *q, const int n){
    float L1 = norm_l1(p, n);
    multiply_scalar(p, 1.f / (L1 + 1e-12f), q, n);
}

void multiply_scalar(const float *p, const float s, float *q, const int n){

    __m128 scale = _mm_set1_ps(s);
    const __m128 *ptr_p = (const __m128*)p;
    __m128 *ptr_q = (__m128*)q;
    int i=0;
    for (; i<(n>>2); i++){
        ptr_q[i] = _mm_mul_ps(ptr_p[i], scale);
    }
    if (n&3)
        for (i=n-(n&3); i<n; i++)
            q[i] = s * p[i];
}


void multiply_dot(const float *p, const float *q, float *z, const int n){


    const __m128 *ptr_p = (const __m128*)p;
    const __m128 *ptr_q = (const __m128*)q;
    __m128 *ptr_z = (__m128*)z;
    int i=0;
    for (; i<(n>>2); i++){
        ptr_z[i] = _mm_mul_ps(ptr_p[i], ptr_q[i]);
    }
    if (n&3)
        for (i=n-(n&3); i<n; i++)
            z[i] = p[i] * q[i];

}


void add(const float *p, const float *q, float *z, const int n){
    const __m128 *ptr_p = (const __m128*)p;
    const __m128 *ptr_q = (const __m128*)q;
    __m128 *ptr_z = (__m128*)z;

    int i=0;
    for (; i<(n>>2); i++){
        ptr_z[i] = _mm_add_ps(ptr_p[i], ptr_q[i]);
    }
    if (n&3)
        for (i=n-(n&3); i<n; i++)
            z[i] = p[i] + q[i];
}

void add_scalar(const float *p, float s, float *z, const int n){

    const __m128 *ptr_p = (const __m128*)p;
    __m128 *ptr_z = (__m128*)z;

    __m128 scale = _mm_set1_ps(s);
    int i=0;
    for (; i<(n>>2); i++){
        ptr_z[i] = _mm_add_ps(ptr_p[i], scale);
    }
    if (n&3)
        for (i=n-(n&3); i<n; i++)
            z[i] = p[i] + s;
}


void subs(const float *p, const float *q, float *z, const int n){
    const __m128 *ptr_p = (const __m128*)p;
    const __m128 *ptr_q = (const __m128*)q;
    __m128 *ptr_z = (__m128*)z;

    int i=0;
    for (; i<(n>>2); i++){
        ptr_z[i] = _mm_sub_ps(ptr_p[i], ptr_q[i]);
    }
    if (n&3)
        for (i=n-(n&3); i<n; i++)
            z[i] = p[i] - q[i];
}


void sqrt(const float *p, float *q, const int n){
    const __m128 *ptr_p = (const __m128*)p;
    __m128 *ptr_q = (__m128*)q;
    unsigned int i=0;

    for (; i< (n>>2); ++i)
        ptr_q[i] = _mm_sqrt_ps(ptr_p[i]);

    if (n & 3)
        for (i= n - (n&3); i<n; ++i)
            q[i] = std::sqrt(p[i]);
}

void dot_div(const float *p, float *q, const int n){

    const __m128 *ptr_p = (const __m128*)p;
    __m128 *ptr_q = (__m128*)q;
    static const __m128 z = {1.f, 1.f, 1.f, 1.f};
    static const __m128 eps = {1e-12f, 1e-12f, 1e-12f, 1e-12f};
    unsigned int i=0;

    for (; i< (n>>2); ++i){
        const __m128 aux = _mm_add_ps(ptr_p[i], eps);
        ptr_q[i] = _mm_div_ps(z, aux);
    }

    if (n & 3)
        for (i= n - (n&3); i<n; ++i)
            q[i] = 1.f / (p[i] + 1e-12f);
}


void threshold(const float *p, const float thr, float *q, const int n){
    const __m128 *ptr_p = (const __m128*)p;
    __m128 *ptr_q = (__m128*)q;
    const __m128 thr_sse = {thr, thr, thr, thr};
    __m128 mask;
    unsigned int i = 0;
    for (; i< (n>>2); ++i){
        mask = _mm_cmpgt_ps(ptr_p[i], thr_sse);
        ptr_q[i] = _mm_or_ps(_mm_and_ps(mask, thr_sse), _mm_andnot_ps(mask, ptr_p[i]));
    }

    if (n & 3)
        for (i= n - (n&3); i<n; ++i)
            q[i] = p[i] > thr ? thr : p[i];
}

float sum(const float *p, const int n){
    float norm;
    float r[4];
    __m128 *ptr_p = (__m128*)p;
    __m128 acc = {0.f, 0.f, 0.f, 0.f};
    int i=0;
    for (; i<n/4; ++i){
        acc = _mm_add_ps(acc, ptr_p[i]);
    }
    _mm_store_ps(r, acc);
    norm = r[0] + r[1] + r[2] + r[3];
    if ( n&3){
        for (i=n-(n&3); i<n; ++i)
            norm += p[i];
    }
    return norm;
}

}
