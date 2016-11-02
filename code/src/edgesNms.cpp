#include "edgesNms.h"
#include <math.h>
#include <omp.h>
using namespace std;

// return I[x,y] via bilinear interpolation
inline float interp( float *I, int h, int w, float x, float y, int len ) {
    x = x<0 ? 0 : (x>w-1.001 ? w-1.001 : x);
    y = y<0 ? 0 : (y>h-1.001 ? h-1.001 : y);
    int x0 = int(x);
    int y0 = int(y);
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    int x1=x0+1;
    int y1=y0+1;
    float dx0=x-x0, dy0=y-y0, dx1=1-dx0, dy1=1-dy0;
    if (x0*h+y0 >= len || x1*h+y0 >= len || x0*h+y1 >= len || x1*h+y1 >= len)
        printf("Oh my god!\n");
    return I[x0*h+y0]*dx1*dy1 + I[x1*h+y0]*dx0*dy1 +
            I[x0*h+y1]*dx1*dy0 + I[x1*h+y1]*dx0*dy0;
}
void edgesNMS(float* E0,float* O,int r, int s, float m, int nThreads, int h, int w, float* E, int len)
{
    // suppress edges where edge is stronger in orthogonal direction
#if USEOMP
    nThreads = nThreads<omp_get_max_threads() ? nThreads : omp_get_max_threads();
#pragma omp parallel for num_threads(nThreads)
#endif
    for( int x=0; x<w; x++ ) for( int y=0; y<h; y++ ) {
        float e=E[x*h+y]=E0[x*h+y]; if(!e) continue; e*=m;
        float coso=cos(O[x*h+y]), sino=sin(O[x*h+y]);
        for( int d=-r; d<=r; d++ ) if( d ) {
            float e0 = interp(E0,h,w,x+d*coso,y+d*sino, len);
            if(e < e0) { E[x*h+y]=0; break; }
        }
    }
    // suppress noisy edge estimates near boundaries
    s=s>w/2?w/2:s; s>h/2? h/2:s;
    for( int x=0; x<s; x++ ) for( int y=0; y<h; y++ ) {
        E[x*h+y]*=x/float(s); E[(w-1-x)*h+y]*=x/float(s); }
    for( int x=0; x<w; x++ ) for( int y=0; y<s; y++ ) {
        E[x*h+y]*=y/float(s); E[x*h+(h-1-y)]*=y/float(s); }
}
