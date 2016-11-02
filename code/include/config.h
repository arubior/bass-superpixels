#ifndef _CONFIG_
#define _CONFIG_


#define LOOPNUMBER 1000

#define USEDOUBLE 0

#if USEDOUBLE
typedef double sType;
#else
typedef float sType;
#endif

#define USEOMP  1
#ifdef USEOMP
#include <omp.h>
#endif
#define PI 3.1415926535897931f
#define MULTISCALE 1

#endif

