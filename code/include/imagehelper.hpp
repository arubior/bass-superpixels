/********************************************
*  Author: Yuheng Chen, chyh1990@gmail.com
*  Date:   19.02.2014
*******************************************/

#ifndef IMAGEHELPER_HPP_8FVB3L0U
#define IMAGEHELPER_HPP_8FVB3L0U

#include <cstdlib>

#ifndef DISABLE_TIMING
#include <ctime>
#include <cstdio>

#ifdef _MSC_VER
#define __TIC__() double __timing_start = clock()
#define __TOC__()                                                 \
    do {                                                      \
        double __timing_end = clock();            \
        fprintf(stdout, "TIME(ms): %lf\n",                \
            (__timing_end - __timing_start)   \
                / CLOCKS_PER_SEC * 1000);         \
    } while (0)
#else
#include <unistd.h>
#include <sys/time.h>

#define __TIC__()                                    \
    struct timeval __timing_start, __timing_end; \
    gettimeofday(&__timing_start, NULL);

#define __TOC__()                                                        \
    do {                                                             \
        gettimeofday(&__timing_end, NULL);                       \
        double __timing_gap = (__timing_end.tv_sec -     \
                           __timing_start.tv_sec) *  \
                          1000.0 +                     \
                      (__timing_end.tv_usec -    \
                           __timing_start.tv_usec) / \
                          1000.0;                    \
        fprintf(stdout, "TIME(ms): %lf\n", __timing_gap);        \
    } while (0)

#endif

#else
#define __TIC__()
#define __TOC__()
#endif

#ifndef _MSC_VER
#define USE_OPENCV
#else
#define USE_GDIPLUS
#endif

#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

#ifdef USE_GDIPLUS
#include <Windows.h>
#pragma comment(lib, "gdiplus.lib")
#include <gdiplus.h>
#endif

namespace sdktest {

class Image {
    public:
    enum ImageType { Image_RGBA, Image_BGR, Image_Gray };
    int cols;
    int rows;
    int stride;
    unsigned char *data;
    ImageType type;
};

#ifdef __APPLE__
struct _ColorQuad {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
};
#else
struct _ColorQuad {
    unsigned char b;
    unsigned char g;
    unsigned char r;
    unsigned char a;
};
#endif

struct ColorQuad : public _ColorQuad {
    ColorQuad(int _b, int _g, int _r, int _a = 255) {
        b = _b; g = _g; r = _r; a = _a;
    }
};

class ImageHelper {
    public:
    virtual Image *LoadRGBAImage(const char *fn) = 0;
    virtual Image *LoadGrayImage(const char *fn) = 0;
    virtual Image *LoadBGRImage(const char *fn) = 0;
    virtual bool SaveImage(const char *fn, const Image *image) = 0;
    virtual Image *CreateImage(int width, int height,
                   Image::ImageType type) {
        Image *img = new Image();
        size_t t = 1;
        if( type == Image::Image_BGR)
            t = 3;
        else if(type == Image::Image_RGBA)
            t = 4;
        img->type = type;
        img->stride = width * t;
        img->rows = height;
        img->cols = width;
        size_t len = img->stride * img->rows;
        img->data = new unsigned char[len];

        return img;
    }

    inline void SetPixel(Image *image, int x, int y, const _ColorQuad &color) {
        if (!image) return;
        if (x < 0 || x >= image->cols
            || y < 0 || y >= image->rows)
            return;
        if (image->type == Image::Image_Gray) {
            image->data[y * image->stride + x] = color.a;
        }
        else if(image->type == Image::Image_BGR){
            int offset = y * image->stride + x * 3;
            unsigned char * p = image->data + offset;
            *p++ = color.b;
            *p++ = color.g;
            *p++ = color.r;
        }
        else {
            /* green */
            int offset = y * image->stride + x * 4;
            *(reinterpret_cast<unsigned int*>(&image->data[offset])) =
                *(reinterpret_cast<const unsigned int*>(&color));
        }
    }

    void DrawPoint(Image *image, int x, int y, int stroke_width,
        const ColorQuad &color = ColorQuad(0, 255, 0)) {
        for (int h = x - stroke_width; h <= x + stroke_width; h++) {
            for (int v = y - stroke_width; v <= y + stroke_width;
                 v++) {
                int dx = h - x, dy = v - y;
                if (dx * dx + dy * dy <=
                    stroke_width * stroke_width)
                    SetPixel(image, h, v, color);
            }
        }
    }
    void DrawRect(Image *image, int top, int left, int right, int bottom,
              int stroke_width,
              const ColorQuad &color = ColorQuad(0, 255, 0)) {
        for (int i = top; i <= bottom; i++) {
            for (int sw = 0; sw < stroke_width; sw++) {
                int u = left + sw;
                int d = right - sw;
                SetPixel(image, u, i, color);
                SetPixel(image, d, i, color);
            }
        }

        for (int i = left; i <= right; i++) {
            for (int sw = 0; sw < stroke_width; sw++) {
                SetPixel(image, i, top + sw, color);
                SetPixel(image, i, bottom - sw, color);
            }
        }
    }
    void DrawMask(Image *image, Image *mask) {
        if (!(image->type == Image::Image_RGBA &&
            mask->type == Image::Image_Gray))
            return;
        for (int r = 0; r < image->rows; r++)
            for (int c = 0; c < image->cols; c++) {
                if (mask->data[mask->stride * r + c])
                    continue;
                *(reinterpret_cast<int *>(image->data
                    + image->stride * r + c * 4)) = 0;
            }
    }
    void FreeImage(Image *image) {
        if (!image) return;
        if (image->data) delete[] image->data;
        delete image;
    }

    virtual ~ImageHelper() {}
};

#ifdef USE_OPENCV

class ImageHelperOpenCV : public ImageHelper {
    public:
    virtual Image *LoadRGBAImage(const char *fn) {
        cv::Mat _img = cv::imread(fn);
        if (!_img.data) return 0;
#ifdef __APPLE__
        cv::cvtColor(_img, _img, CV_BGR2RGBA);
#else
        cv::cvtColor(_img, _img, CV_BGR2BGRA);
#endif

        Image *img =
            CreateImage(_img.cols, _img.rows, Image::Image_RGBA);
        size_t len = _img.step * _img.rows;
        memcpy(img->data, _img.data, len);

        return img;
    }

    virtual Image *LoadGrayImage(const char *fn) {
        cv::Mat _img = cv::imread(fn);
        if (!_img.data) return 0;
        cv::cvtColor(_img, _img, CV_BGR2GRAY);

        Image *img =
            CreateImage(_img.cols, _img.rows, Image::Image_Gray);
        size_t len = _img.step * _img.rows;

        memcpy(img->data, _img.data, len);

        return img;
    }

    Image *LoadBGRImage(const char *fn){
        cv::Mat _img = cv::imread(fn);
        if (!_img.data) return 0;

        Image *img =
            CreateImage(_img.cols, _img.rows, Image::Image_BGR);
        size_t len = _img.step * _img.rows;
        memcpy(img->data, _img.data, len);

        return img;
    }

    virtual bool SaveImage(const char *fn, const Image *image) {
        if (!image) return false;
        int type = CV_8UC1;
        if(image->type == Image::Image_BGR)
            type = CV_8UC3;
        else if(image->type == Image::Image_RGBA)
            type = CV_8UC4;
        cv::Mat mat(image->rows, image->cols,
                type,
                image->data);
#ifdef __APPLE__
        if(image->type == Image_RGBA)
            cv::cvtColor(mat, mat, CV_RGBA2BGRA);
#endif
        cv::imwrite(fn, mat);
        return true;
    }
};

#define ImageHelperBackend ImageHelperOpenCV
#endif
/* GDIPLUS store pixel in BGRA format (same as opencv) */
#ifdef USE_GDIPLUS
class ImageHelperGdiPlus : public ImageHelper {
    private:
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    ULONG_PTR gdiplusToken;
    int GetEncoderClsid(const WCHAR *format, CLSID *pClsid) {
        UINT num = 0;   // number of image encoders
        UINT size = 0;  // size of the image encoder array in bytes

        Gdiplus::ImageCodecInfo *pImageCodecInfo = NULL;

        Gdiplus::GetImageEncodersSize(&num, &size);
        if (size == 0) return -1;  // Failure

        pImageCodecInfo = (Gdiplus::ImageCodecInfo *)(malloc(size));
        if (pImageCodecInfo == NULL) return -1;  // Failure

        GetImageEncoders(num, size, pImageCodecInfo);

        for (UINT j = 0; j < num; ++j) {
            if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
                *pClsid = pImageCodecInfo[j].Clsid;
                free(pImageCodecInfo);
                return j;  // Success
            }
        }

        free(pImageCodecInfo);
        return -1;  // Failure
    }

    public:
    ImageHelperGdiPlus() {
        Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput,
                    NULL);
    }
    ~ImageHelperGdiPlus() {
        Gdiplus::GdiplusShutdown(gdiplusToken);
    }

    virtual Image *LoadRGBAImage(const char *fn) {
        WCHAR path[2048];
        size_t wchar_len = 0;
        mbstowcs_s(&wchar_len, path, fn, 2048);
        Gdiplus::Bitmap *_img = Gdiplus::Bitmap::FromFile(path);
        if (!_img) return 0;

        Image *img = CreateImage(_img->GetWidth(), _img->GetHeight(),
                     Image::Image_RGBA);
        Gdiplus::BitmapData _data;
        Gdiplus::Rect _rect(0, 0, _img->GetWidth(), _img->GetHeight());
        if (Gdiplus::Ok ==
            _img->LockBits(&_rect, Gdiplus::ImageLockModeRead,
                   PixelFormat32bppARGB, &_data)) {
            BYTE *p = reinterpret_cast<BYTE *>(_data.Scan0);
            BYTE *pd = img->data;
            memcpy(pd, p, 4 * _img->GetWidth() * _img->GetHeight());
            _img->UnlockBits(&_data);
        } else {
            FreeImage(img);
            delete _img;
            return 0;
        }
        delete _img;
        return img;
    }

    virtual Image *LoadBGRImage(const char *fn){
        return NULL;
    }

    virtual Image *LoadGrayImage(const char *fn) {
        using namespace Gdiplus;
        WCHAR path[2048];
        size_t wchar_len = 0;
        mbstowcs_s(&wchar_len, path, fn, 2048);
        Bitmap *image = Gdiplus::Bitmap::FromFile(path);
        if (!image) {
            // *img = NULL;
            return NULL;
        }
        BitmapData _data;
        Rect _rect(0, 0, image->GetWidth(), image->GetHeight());
        Image *img = CreateImage(image->GetWidth(), image->GetHeight(),
                     Image::Image_Gray);
        BYTE *gray = img->data;
#define GREY(r, g, b) ((((r)*77) + ((g)*151) + ((b)*28)) >> 8)
        if (Ok ==
            image->LockBits(&_rect,
                    ImageLockModeRead,
                    PixelFormat32bppARGB, &_data)) {
            BYTE *p = reinterpret_cast<BYTE *>(_data.Scan0);
            BYTE *pd = gray;
            for (UINT i = 0; i < _data.Height; i++) {
                for (UINT j = 0; j < _data.Width; j++) {
                    *pd++ = GREY(p[4 * j + 2], p[4 * j + 1],
                             p[4 * j + 0]);
                }
                p += _data.Stride;
            }
            image->UnlockBits(&_data);
        } else {
            FreeImage(img);
            delete image;
            return NULL;
        }
#undef GREY
        delete image;
        return img;
    }

    virtual bool SaveImage(const char *fn, const Image *image) {
        if (!image) return false;
        WCHAR path[1024];
        mbstowcs(path, fn, 1024);
        CLSID pngClsid;
        GetEncoderClsid(L"image/jpeg", &pngClsid);
        Gdiplus::Bitmap _img(image->cols, image->rows,
                     PixelFormat32bppARGB);
        Gdiplus::BitmapData _data;
        Gdiplus::Rect _rect(0, 0, _img.GetWidth(), _img.GetHeight());
        if (Gdiplus::Ok ==
            _img.LockBits(&_rect, Gdiplus::ImageLockModeWrite,
                  PixelFormat32bppARGB, &_data)) {
            BYTE *p = reinterpret_cast<BYTE *>(_data.Scan0);
            BYTE *pd = image->data;
            memcpy(p, pd, 4 * _img.GetWidth() * _img.GetHeight());
            _img.UnlockBits(&_data);
            _img.Save(path, &pngClsid);
        } else {
            return false;
        }
        return true;
    }
};

#define ImageHelperBackend ImageHelperGdiPlus
#endif
}  // namespace sdktest

#endif /* end of include guard: IMAGEHELPER_HPP_8FVB3L0U */
