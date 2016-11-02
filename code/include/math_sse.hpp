#ifndef _MATH_SSE_H_
#define _MATH_SSE_H_

#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) & (BYTE_COUNT) == 0)

namespace MathSSE{

float norm_l1(const float *p, const int n);

void normalize_l1(const float *p, float *q, const int n);

void multiply_scalar(const float *p, const float s, float *q, const int n);

void multiply_dot(const float *p, const float *q, float *z, const int n);

void add(const float *p, const float *q, float *z, const int n);

void add_scalar(const float *p, float s, float *z, const int n);

void subs(const float *p, const float *q, float *z, const int n);

void sqrt(const float *p, float *q, const int n);

// q[i] = 1 / (p[i] + eps)
void dot_div(const float *p, float *q, const int n);

void threshold(const float *p, const float thr, float *q, const int n);

float sum(const float *p, const int n);
}
#endif
