#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "Tools.h"
#include <string>
#include <eigen3/Eigen/Core> // For RMatrixXf (geodesic distance)
#include <opencv2/core/eigen.hpp>
#include <iostream>
#include <stack>
#include <ctime>
#include <dirent.h>
#include <cstdlib> 
#include <unistd.h>
#include <sstream>
// #include "structureEdge.h" // Uncomment for edge detection

using namespace cv;
using namespace std;

stack<clock_t> tictoc_stack;

void tic() {
    tictoc_stack.push(clock());
}

void toc() {
    std::cout << "Time elapsed: "
    << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC << " seconds"
    << std::endl;
    tictoc_stack.pop();
}

// void mat2uchar(_matrix<unsigned char>& img, Mat t)
// {
    // int size[3];
    // size[0] = t.rows;size[1] = t.cols;size[2] = 3;
    // img.reShape(3,size);
// 
    // int* th = new int[size[1]];
    // int* tw = new int[size[0]];
// 
    // for (int i = 0; i < size[1]; i++) th[i] = i*size[0];
    // for (int i = 0; i < size[0]; i++) tw[i] = i*size[1]*3;
// 
    // for (int k = 0; k < size[2];k++){
        // int ind1 = size[0] * size[1] * k;
        // for (int j = 0; j < size[1];j++){
            // int j3 = j * 3;
            // for (int i = 0; i < size[0];i++){
                // img.value[ind1 + th[j] + i] = t.data[tw[i] + j3 + 2 - k];
            // }
        // }
    // }
    // delete [] th;
    // delete [] tw;
// } 

void edgeDetection2(Mat image, Mat &thEdges)
{
    Mat src, src_gray;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    GaussianBlur( image, src, Size(3,3), 0, 0, BORDER_DEFAULT );

    /// Convert image to gray
    // cvtColor( src, src_gray, CV_BGR2GRAY ); // WE
    cvtColor( src, src_gray, cv::COLOR_BGR2GRAY ); // IRI

    /// Generate grad_x and grad_y
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, thEdges );

    // Prepare for thresholding:
    thEdges.convertTo(thEdges,CV_32FC1);
    thEdges = thEdges/255;

    threshold( thEdges, thEdges, 0.4, 1, THRESH_BINARY);
}

/*
void edgeDetection(Mat image, Mat &thEdges, Model* model)
{
        _matrix<unsigned char> img_;
        _matrix<float> edgesImg;

        mat2uchar(img_, image);
        structureEdge2(img_,*model,edgesImg);

        Mat rotEdges(edgesImg.size[1], edgesImg.size[0], CV_32FC1, edgesImg.value);
        Mat edgesMat;

        transpose(rotEdges,edgesMat);

        threshold( edgesMat, thEdges, 0.3, 1, THRESH_BINARY);

        // imshow( "Th_edges", thEdges);
        // waitKey(0);
        // thEdges = thEdges/255;
        // imwrite("/home/wideeyes/Desktop/edges.png", thEdges);
}
*/

void rgbtolab(int* rin, int* gin, int* bin, int sz, double* lvec, double* avec, double* bvec)
{
    int i; int sR, sG, sB;
    double R,G,B;
    double X,Y,Z;
    double r, g, b;
    const double epsilon = 0.008856;    //actual CIE standard
    const double kappa   = 903.3;       //actual CIE standard
    
    const double Xr = 0.950456; //reference white
    const double Yr = 1.0;      //reference white
    const double Zr = 1.088754; //reference white
    double xr,yr,zr;
    double fx, fy, fz;
    double lval,aval,bval;
    
    for(i = 0; i < sz; i++)
    {
        sR = rin[i]; sG = gin[i]; sB = bin[i];
        R = sR/255.0;
        G = sG/255.0;
        B = sB/255.0;
        
        if(R <= 0.04045)    r = R/12.92;
        else                r = pow((R+0.055)/1.055,2.4);
        if(G <= 0.04045)    g = G/12.92;
        else                g = pow((G+0.055)/1.055,2.4);
        if(B <= 0.04045)    b = B/12.92;
        else                b = pow((B+0.055)/1.055,2.4);
        
        X = r*0.4124564 + g*0.3575761 + b*0.1804375;
        Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
        Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
        
        //------------------------
        // XYZ to LAB conversion
        //------------------------
        xr = X/Xr;
        yr = Y/Yr;
        zr = Z/Zr;
        
        if(xr > epsilon)    fx = pow(xr, 1.0/3.0);
        else                fx = (kappa*xr + 16.0)/116.0;
        if(yr > epsilon)    fy = pow(yr, 1.0/3.0);
        else                fy = (kappa*yr + 16.0)/116.0;
        if(zr > epsilon)    fz = pow(zr, 1.0/3.0);
        else                fz = (kappa*zr + 16.0)/116.0;
        
        lval = 116.0*fy-16.0;
        aval = 500.0*(fx-fy);
        bval = 200.0*(fy-fz);
        
        lvec[i] = lval; avec[i] = aval; bvec[i] = bval;
    }
}

void addDeleteSeeds(int* xseeds, int* yseeds, int* numseeds, int numSuperpixels, Mat intImg, Mat image, int* ndel){

    double nrows = intImg.rows-1; // Integral image adds
    double ncols = intImg.cols-1; //  one row and column
    double lh = nrows/(nrows + ncols);
    double lw = ncols/(nrows + ncols);
    double nh = round(2*sqrt(numSuperpixels)*lh);
    double nw = round(2*sqrt(numSuperpixels)*lw);
    double ih = round(nrows/nh);
    double iw = round(ncols/nw);
    int deleted = 0;
    int n = 0;
    int xdiff = 0;
    int ydiff = 0;
    int* cornersx;
    int* cornersy;
    cornersx = new int [4];
    cornersy = new int [4];
    for (int ci = 0; ci < 4; ci++)
    {
        cornersx[ci] = 0;
        cornersy[ci] = 0;
    }

    int* yb = new int [(int)nh];
    int* xb = new int [(int)nw];

    for (int k = 0; k < nh; k++)
    {
        yb[k] = round(ih/2) + k*ih;
    }
    for (int l = 0; l < nw; l++)
    {
        xb[l] = round(iw/2) + l*iw;
    }

    Mat xbMat(Size(nw,1),CV_32S,xb);
    Mat x(nw*nh,1,CV_8UC3,Scalar(0));
    repeat(xbMat,1,nh,x);

    Mat ybMat(Size(1,nh),CV_32S,yb);
    Mat y_(nw,nh,CV_8UC3,Scalar(0));
    repeat(ybMat,1,nw,y_);
    
    // transpose(y_,y_2);
    Mat y = y_.reshape(0,1);

    double ii_th = intImg.at<double>(nrows,ncols)/numSuperpixels;
    
    int ns = 0;
    bool TO_ADD;
    transpose(x,x); // To use Mat.push_back(int)
    transpose(y,y); // To use Mat.push_back(int)
    int lenXY = nh*nw;

    int* tmpx = new int [lenXY*5]; // Maximum size = seeds + 4 corners per seed
    int* tmpy = new int [lenXY*5]; // Maximum size = seeds + 4 corners per seed
    for (int tmpi = 0; tmpi < lenXY*5; tmpi++)
    {
        if(tmpi < lenXY)
        {
            // tmpx[tmpi] = x.at<int>(1,tmpi);
            // tmpy[tmpi] = y.at<int>(1,tmpi);    
            tmpx[tmpi] = x.at<int>(tmpi);
            tmpy[tmpi] = y.at<int>(tmpi);    
        }
        else
        {
            tmpx[tmpi] = 0;
            tmpy[tmpi] = 0;    
        }
        // printf("%i => (%i, %i)\n", tmpi, tmpx[tmpi], tmpy[tmpi]);
        
    }

    // for (int lXY = 0; lXY < lenXY; lXY++)
    // {
    // xseeds[lXY] = tmpx[lXY];
    // yseeds[lXY] = tmpy[lXY];
    // }
    // *numseeds = lenXY;
    // *ndel = deleted;

    // // Show seeds:
    // Mat seedsImg2(image.rows, image.cols, CV_32S, Scalar(0));
    // image.copyTo(seedsImg2);

    /////////////// SHOW SEEDS /////////////////
    // Mat seedsImg3;
    // image.copyTo(seedsImg3);
    // for (int p = 0; p < lenXY; p++)
    // {
    //     int i = tmpx[p];
    //     int j = tmpy[p];
    //     Point center = {i,j};
    //     int radius = 3;
    //     const Scalar rgb = 100;
    //     circle(seedsImg3, center ,radius, rgb,2);
    // }
    // imshow("Initial seeds", seedsImg3);
    // waitKey(0);
    // imwrite("/home/wideeyes/Desktop/Initial_seeds.png",seedsImg3);



    for (int c = 0; c < nh*nw; c++)
    {
        cornersx[0] = x.at<int>(c) + round(iw/2); 
        cornersy[0] = y.at<int>(c) + round(ih/2);
        cornersx[1] = x.at<int>(c) - round(iw/2); 
        cornersy[1] = y.at<int>(c) - round(ih/2);
        cornersx[2] = x.at<int>(c) + round(iw/2); 
        cornersy[2] = y.at<int>(c) - round(ih/2);
        cornersx[3] = x.at<int>(c) - round(iw/2); 
        cornersy[3] = y.at<int>(c) + round(ih/2);


        for (int cc = 0; cc < 4; cc++)
        {
            if(cornersx[cc] < 0){cornersx[cc] = 0;}
            if(cornersy[cc] < 0){cornersy[cc] = 0;}
            if(cornersx[cc] >= ncols-1){cornersx[cc] = ncols - 1;}
            if(cornersy[cc] >= nrows-1){cornersy[cc] = nrows - 1;}
            n++;
        }

        double sumArea;

        int tl= intImg.at<double>(cornersy[1],cornersx[1]);
        int tr= intImg.at<double>(cornersy[2],cornersx[2]);
        int bl= intImg.at<double>(cornersy[3],cornersx[3]);
        int br= intImg.at<double>(cornersy[0],cornersx[0]); 

        sumArea = br-bl-tr+tl;

        bool DELETE_SEED = (sumArea < ii_th);
        bool ADD_CORNERS = (sumArea > 3*ii_th);

        if (!DELETE_SEED)
        {
            xseeds[ns] = tmpx[c];
            yseeds[ns] = tmpy[c];            

            // xseeds[ns] = x.at<int>(1,c);
            // yseeds[ns] = y.at<int>(1,c);
            
            ns++;

            ///////////////// Remove duplicate seeds:  /////////////////
            for (int rd = 0; rd < 4; rd++)
            {
                TO_ADD = true;
                for (int rds = nh*nw; rds < lenXY; rds++)
                { 
                    xdiff = abs(cornersx[rd]-tmpx[rds]);
                    ydiff = abs(cornersy[rd]-tmpy[rds]);
                    // xdiff = abs(cornersx[rd]-x.at<int>(1,rds));
                    // ydiff = abs(cornersy[rd]-y.at<int>(1,rds));
                    if (ydiff <= 1 && xdiff <= 1)
                    {
                        TO_ADD = false;
                    }
                }

                if (ADD_CORNERS && TO_ADD)
                {
                    // x.push_back(cornersx[rd]);
                    // y.push_back(cornersy[rd]);
                    tmpx[lenXY-1] = cornersx[rd];
                    tmpy[lenXY-1] = cornersy[rd];
                    lenXY++;
                    xseeds[ns] = cornersx[rd];
                    yseeds[ns] = cornersy[rd];
                    ns++;
                }
            }
        }
        else
        {
            deleted++;
        }

    }


    Mat z(intImg.rows, intImg.cols, CV_8UC1, Scalar(0));

    int* cornerszx = new int [2];
    int* cornerszy = new int [2];
    for (int cixy = 0; cixy < 2; cixy++)
    {
        cornerszx[cixy] = 0;
        cornerszy[cixy] = 0;
    }

    for(int l = 0; l < ns; l++)
    {
        cornerszx[0] = xseeds[l] + round(iw/2)+1;
        cornerszy[0] = yseeds[l] + round(ih/2)+1;
        cornerszx[1] = xseeds[l] - round(iw/2)+1;
        cornerszy[1] = yseeds[l] - round(ih/2)+1;

        for (int cc = 0; cc < 2; cc++)
        {
            if(cornerszx[cc] < 0){cornerszx[cc] = 0;}
            if(cornerszy[cc] < 0){cornerszy[cc] = 0;}
            if(cornerszx[cc] >= ncols){cornerszx[cc] = ncols;}
            if(cornerszy[cc] >= nrows){cornerszy[cc] = nrows;}
        }

        Mat zAux = z.colRange(cornerszx[1],cornerszx[0]).rowRange(cornerszy[1],cornerszy[0]);
        Mat Ones(zAux.rows, zAux.cols, CV_8UC1, Scalar(1));
        Ones.copyTo(zAux);
    }

    bitwise_not(255*z,z);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    // WE:
    // findContours(z, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    // IRI:
    findContours(z, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    // RNG rng(12345);

    // Mat drawing = Mat::zeros( z.size(), CV_8UC3 );
    // for( int i = 0; i< contours.size(); i++ )
    //  {
    //    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
    //    drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
    //  }

    Moments mu;
    Point2f mc;

    for( int i = 0; i < contours.size(); i++ )
    {
    //  Find the area of contour
        double a=contourArea( contours[i],false); 
        if(a>ih*iw){
            mu = moments( contours[i], false );
            mc = Point2f( mu.m10/mu.m00 , mu.m01/mu.m00 );

            xseeds[ns] = (int)round(mc.x);
            yseeds[ns] = (int)round(mc.y);
            ns++;
            lenXY++;
        }
    }

    ///////////////// SHOW SEEDS /////////////////
    // Mat seedsImg2;
    // image.copyTo(seedsImg2);
    // for (int p = 0; p < lenXY; p++)
    // {
    //     int i = xseeds[p];
    //     int j = yseeds[p];
    //     Point center = {i,j};
    //     int radius = 3;
    //     const Scalar rgb = 100;
    //     circle(seedsImg2, center ,radius, rgb, 2);
    // }
    // imshow("Final seeds", seedsImg2);
    // waitKey(0);
    // imwrite("/home/wideeyes/Desktop/Final_seeds.png",seedsImg2);

    *numseeds = ns;
    *ndel = deleted;

    delete [] cornersx;
    delete [] cornersy;
    delete [] yb;
    delete [] xb;
    delete [] cornerszx;
    delete [] cornerszy;
    delete [] tmpx;
    delete [] tmpy;
}

void DTOCS(Mat src_img, Mat &dst, bool &on_edge, int seedx, int seedy)
{
    if(src_img.at<float>(seedy,seedx) == 1)
    {
        on_edge = true;
        return;
    }
    else
    {
        on_edge = false;
    }

    if(seedy < 0){seedy = 0;}
    if(seedx < 0){seedx = 0;}
    if(seedy >= dst.rows){seedy = dst.rows-1;}
    if(seedx >= dst.cols){seedx = dst.cols-1;}
    dst.at<float>(seedy,seedx) = 0;

    float wt = 100;
    int niters = 2;

    int r = 0;
    int c1 = 0;
    int c2 = 0;
    float va = 0;
    float vb = 0;
    float vc = 0;
    float vd = 0;
    float vf = 0;
    float vg = 0;
    float vh = 0;
    float vk = 0;
    float mindist = 0;
    double tmp = 0;

    vector<float> vec(4);
    for (int itv = 0; itv < 4; itv++){vec[itv] = 0;}

    for (int it = 0; it < niters; it++)
    {
        // First iteration:
        for (int i = 0; i < src_img.rows; i++)
        {
            for (int j = 0; j < src_img.cols; j++)
            {
                r = i -1; 
                c1 = j-1;
                c2 = j+1;

                if (r<0 || c1 < 0){va = FLT_MAX;}
                else
                {
                    tmp = wt*abs(src_img.at<float>(i,j) - src_img.at<float>(i-1,j-1)) + dst.at<float>(i-1,j-1);
                    va = 1 + tmp;
                    if(tmp == FLT_MAX){va = 0;}
                }

                if (r<0){vb = FLT_MAX;}
                else
                {
                    tmp = wt*abs(src_img.at<float>(i,j) - src_img.at<float>(i-1,j)) + dst.at<float>(i-1,j);                    
                    vb = 1 + tmp;
                    if(tmp == FLT_MAX){vb = 0;}
                }

                if (r<0 || c2 >= src_img.cols){vc = FLT_MAX;}
                else
                {
                    tmp = wt*abs(src_img.at<float>(i,j) - src_img.at<float>(i-1,j+1)) + dst.at<float>(i-1,j+1);                    
                    vc = 1 + tmp;
                    if(tmp == FLT_MAX){vc = 0;}
                }

                if (c1 < 0){vd = FLT_MAX;}
                else
                {
                    tmp = wt*abs(src_img.at<float>(i,j) - src_img.at<float>(i,j-1)) + dst.at<float>(i,j-1);                    
                    vd = 1 + tmp;
                    if(tmp == FLT_MAX){vd = 0;}
                }

                vec[0] = va;
                vec[1] = vb; 
                vec[2] = vc;
                vec[3] = vd;
                mindist = *min_element(vec.begin(), vec.end());
                dst.at<float>(i,j) = min(dst.at<float>(i,j),mindist);
            }
        }

        // Second iteration:
        for (int i = src_img.rows -1; i > -1; i--)
        {
            for (int j = src_img.cols -1; j > -1; j--)
            {
                r = i + 1;
                c1 = j-1;
                c2 = j+1;
                
                if (c2 >= src_img.cols){vf = FLT_MAX;}
                else
                {
                    tmp = wt*abs(src_img.at<float>(i,j) - src_img.at<float>(i,j+1)) + dst.at<float>(i,j+1);                    
                    vf = 1 + tmp;
                    if(tmp == FLT_MAX){vf = 0;}
                }

                if (r >= src_img.rows || c1 < 0){vg = FLT_MAX;}
                else
                {
                    tmp = wt*abs(src_img.at<float>(i,j) - src_img.at<float>(i+1,j-1)) + dst.at<float>(i+1,j-1);
                    vg = 1 + tmp;
                    if(tmp == FLT_MAX){vg = 0;}
                }

                if (r >= src_img.rows){vh = FLT_MAX;}
                else
                {
                    tmp = wt*abs(src_img.at<float>(i,j) - src_img.at<float>(i+1,j)) + dst.at<float>(i+1,j);
                    vh = 1 + tmp;
                    if(tmp == FLT_MAX){vh = 0;}
                }

                if (r >= src_img.rows || c2 >= src_img.cols){vk = FLT_MAX;}
                else
                {
                    tmp = wt*abs(src_img.at<float>(i,j) - src_img.at<float>(i+1,j+1)) + dst.at<float>(i+1,j+1);
                    vk = 1 + tmp;
                    if(tmp == FLT_MAX){vk = 0;}
                }

                vec[0] = vf;
                vec[1] = vg; 
                vec[2] = vh;
                vec[3] = vk;

                mindist = *min_element(vec.begin(), vec.end());
                dst.at<float>(i,j) = min(dst.at<float>(i,j),mindist);
            }
        }
    }
}

void PerformBASS(double* lvec, double* avec, double* bvec, double* kseedsl, double* kseedsa, double* kseedsb, double* kseedsx, double* kseedsy, int width, int height, int numseeds, int* klabels, int STEP, double compactness, Mat thEdges, double wtgeo, double th, double invwt, bool POSWT, double* pLAB, double* pXY, double* pG)
{
    int x1, y1, x2, y2;
    double l, a, b;
    double dist;
    double distlab;
    double distxy;
    double distGeo;
    int itr;
    int n;
    int x,y;
    int i;
    int ind;
    int r,c;
    int k;
    int sz = width*height;
    int offset = 2*STEP;
    double wtgeo2 = wtgeo;
    
    double* clustersize = new double [numseeds];
    double* inv         = new double [numseeds];
    double* sigmal      = new double [numseeds];
    double* sigmaa      = new double [numseeds];
    double* sigmab      = new double [numseeds];
    double* sigmax      = new double [numseeds];
    double* sigmay      = new double [numseeds];
    double* distvec     = new double [sz];

    for (int ii = 0; ii < numseeds; ii++)
    {
        clustersize[ii] = 0;
        inv[ii] = 0;
        sigmal[ii] = 0;
        sigmaa[ii] = 0;
        sigmab[ii] = 0;
        sigmax[ii] = 0;
        sigmay[ii] = 0;
    }

    for (int ij = 0; ij < sz; ij++)
    {
        distvec[ij] = 0;
    }

    if (!POSWT)
        invwt = 1.0/((STEP/compactness)*(STEP/compactness));

    // For Geodesic distance:
    float sca = 0.5;
    Mat trEdges(thEdges.rows, thEdges.cols, CV_32S, Scalar(0));
    thEdges.copyTo(trEdges);
    resize(trEdges, trEdges, Size(), sca, sca, INTER_NEAREST);

    double maxG = 0;

    // Create a structuring element
    int erosion_size = 1;  
    Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1), cv::Point(erosion_size, erosion_size) );
    dilate(trEdges,trEdges,element);

    double Mdist = 0;
    double Mdistlab = 0;
    double Mdistxy = 0;
    double MdistGeo = 0;
    int count = 0;

    double MMdist = 0;
    double MMdistlab = 0;
    double MMdistxy = 0;
    double MMdistGeo = 0;
    int Mcount = 0;
    bool on_edge = false;

    for( itr = 0; itr < 10; itr++ )
    {
    // Mat geoDistMat(trEdges.rows, trEdges.cols, CV_32FC1, Scalar(0));
        Mdist = 0;
        Mdistlab = 0;
        Mdistxy = 0;
        MdistGeo = 0;
        count = 0;

        if(th==0){for(i=0; i < sz; i++){distvec[i] = DBL_MAX;}}
        else{for(i=0; i < sz; i++){distvec[i] = th;}}

        for( n = 0; n < numseeds; n++ )
        {
            x1 = kseedsx[n]-offset; if(x1 < 0) x1 = 0;
            y1 = kseedsy[n]-offset; if(y1 < 0) y1 = 0;
            x2 = kseedsx[n]+offset; if(x2 > width)  x2 = width - 1;
            y2 = kseedsy[n]+offset; if(y2 > height) y2 = height -1;

            int x_r = floor(sca*x1),
                y_r = floor(sca*y1), 
                width_r = round(sca*x2)-round(sca*x1),
                height_r = round(sca*y2)-round(sca*y1);

            int Xs = round(sca*kseedsx[n]) - x_r;
            int Ys = round(sca*kseedsy[n]) - y_r;

            Mat edgesROI = trEdges(Rect(x_r, y_r, width_r, height_r));
            
            Mat geoDistMat(trEdges.rows, trEdges.cols, CV_32FC1, Scalar(0));
            Mat aux_geoDistMat(edgesROI.rows, edgesROI.cols, CV_32FC1, Scalar(FLT_MAX));


            if(wtgeo != 0)
            {
                // DTOCS(trEdges, geoDistMat, on_edge, round(sca*kseedsx[n]),round(sca*kseedsy[n]));
                DTOCS(edgesROI, aux_geoDistMat, on_edge, Xs,Ys);
                if( on_edge ){
                    wtgeo2 = 0;
                }
                else
                {
                    maxG = *max_element(aux_geoDistMat.begin<float>(),aux_geoDistMat.end<float>());
                    aux_geoDistMat = aux_geoDistMat/maxG;
                    GaussianBlur( aux_geoDistMat, aux_geoDistMat, Size(9,9), 0, 0, BORDER_DEFAULT );
                    wtgeo2 = wtgeo;

                    Mat dst_roi = geoDistMat(Rect(x_r, y_r, width_r, height_r));
                    aux_geoDistMat.copyTo(dst_roi);


                    ///////////////// SHOW SEEDS /////////////////
                    // Mat seedsImg2;
                    // geoDistMat.copyTo(seedsImg2);
                    //     int i = round(sca*kseedsx[n]);
                    //     int j = round(sca*kseedsy[n]);
                    //     Point center = {i,j};
                    //     int radius = 3;
                    //     const Scalar rgb = 100;
                    //     circle(seedsImg2, center ,radius, rgb);
                    // imshow("z", seedsImg2);
                    // waitKey(0);

                    // std::string n_str = std::to_string(n);

                    // string storegeo = "/home/wideeyes/Desktop/DTOCS/dtocs_" + n_str + ".png";
                    // imwrite(storegeo,seedsImg2);

                    // imshow("Geo", geoDistMat);
                    // waitKey(0);
                }
            }

            for( y = y1; y < y2; y++ )
            {
                for( x = x1; x < x2; x++ )
                {
                    i = y*width + x;

                //////////// COLOR DISTANCE ////////////
                    l = lvec[i];
                    a = avec[i];
                    b = bvec[i];

                    distlab =       (l - kseedsl[n])*(l - kseedsl[n]) +
                                    (a - kseedsa[n])*(a - kseedsa[n]) +
                                    (b - kseedsb[n])*(b - kseedsb[n]);

                //////////// EUCLIDEAN DISTANCE ////////////
                    distxy =        (x - kseedsx[n])*(x - kseedsx[n]) + (y - kseedsy[n])*(y - kseedsy[n]);

                //////////// GEODESIC DISTANCE ////////////
                    if (wtgeo2 != 0)
                    {
                        int ry = round(sca*y);
                        int rx = round(sca*x);
                        if (ry < 0){ry = 0;}
                        if (ry > geoDistMat.rows - 1){ry = geoDistMat.rows-1;}
                        if (rx < 0){rx = 0;}
                        if (rx > geoDistMat.cols - 1){rx = geoDistMat.cols-1;}

                        distGeo =       geoDistMat.at<float>(ry,rx);
                    }
                    else
                    {
                        distGeo = 0;
                    }

                //////////// TOTAL DISTANCE ////////////
                    dist = distlab + distxy*invwt + wtgeo*distGeo;

                    Mdist += dist;
                    Mdistlab += distlab;
                    Mdistxy += distxy;
                    MdistGeo += distGeo;
                    count++;

                    if(dist < distvec[i])
                    {
                        distvec[i] = dist;
                        klabels[i]  = n;
                    }
                }
            }
        }

        MMdist += Mdist/(double)count;
        MMdistlab += Mdistlab/(double)count;
        MMdistxy += invwt*Mdistxy/(double)count;
        MMdistGeo += wtgeo*MdistGeo/(double)count;
        Mcount++;

        //-----------------------------------------------------------------
        // Recalculate the centroid and store in the seed values
        //-----------------------------------------------------------------
        for(k = 0; k < numseeds; k++)
        {
            sigmal[k] = 0;
            sigmaa[k] = 0;
            sigmab[k] = 0;
            sigmax[k] = 0;
            sigmay[k] = 0;
            clustersize[k] = 0;
        }

        ind = 0;
        for( r = 0; r < height; r++ )
        {
            for( c = 0; c < width; c++ )
            {
                if(klabels[ind] >= 0)
                {
                    sigmal[klabels[ind]] += lvec[ind];
                    sigmaa[klabels[ind]] += avec[ind];
                    sigmab[klabels[ind]] += bvec[ind];
                    sigmax[klabels[ind]] += c;
                    sigmay[klabels[ind]] += r;
                    clustersize[klabels[ind]] += 1.0;
                }
                ind++;
            }
        }

        for( k = 0; k < numseeds; k++ )
        {
            if( clustersize[k] <= 0 ) clustersize[k] = 1;
            inv[k] = 1.0/clustersize[k];//computing inverse now to multiply, than divide later
        }
    
        for( k = 0; k < numseeds; k++ )
        {
            kseedsl[k] = sigmal[k]*inv[k];
            kseedsa[k] = sigmaa[k]*inv[k];
            kseedsb[k] = sigmab[k]*inv[k];
            kseedsx[k] = sigmax[k]*inv[k];
            kseedsy[k] = sigmay[k]*inv[k];
        }

    }

    // cout << "Total Mean dist = " << MMdist/(double)itr << "\t Total Mean LAB = " << MMdistlab/(double)itr << "\t Total Mean XY = " << MMdistxy/(double)itr << "\t Total Mean Geo = " << MMdistGeo/(double)itr << endl;
    cout << "dist = " << (MMdist/(double)itr)/(MMdist/(double)itr)*100 << " => %\t LAB = " << (MMdistlab/(double)itr)/(MMdist/(double)itr)*100 << "%\t XY = " << (MMdistxy/(double)itr)/(MMdist/(double)itr)*100 << "%\t Geo = " << (MMdistGeo/(double)itr)/(MMdist/(double)itr)*100 << "%" << endl;

    *pLAB = (MMdistlab/(double)itr)/(MMdist/(double)itr)*100; 
    *pXY =  (MMdistxy/(double)itr)/(MMdist/(double)itr)*100;
    *pG =   (MMdistGeo/(double)itr)/(MMdist/(double)itr)*100;

    delete[] sigmal;
    delete[] sigmaa;
    delete[] sigmab;
    delete[] sigmax;
    delete[] sigmay;
    delete[] clustersize;
    delete[] inv;
    delete[] distvec;
}


void EnforceSuperpixelConnectivity(int* labels, int width, int height, int numSuperpixels,int* nlabels, int* finalNumberOfLabels, double maxSUPSZ)
{
    int i,j,k;
    int n,c,count;
    int x,y;
    int ind;
    int oindex, adjlabel;
    int label;
    const int dx4[4] = {-1,  0,  1,  0};
    const int dy4[4] = { 0, -1,  0,  1};
    const int sz = width*height;
    const int SUPSZ = sz/numSuperpixels;
    int* xvec; 
    int* yvec; 
    xvec = new int [SUPSZ*50]; 
    yvec = new int [SUPSZ*50]; 
    for( i = 0; i < sz; i++ ) nlabels[i] = -1;
    oindex = 0; 
    adjlabel = 0;//adjacent label
    label = 0;
    for( j = 0; j < height; j++ )
    {
        for( k = 0; k < width; k++ )
        {
            if( 0 > nlabels[oindex] )
            {
                nlabels[oindex] = label;
                //--------------------
                // Start a new segment
                //--------------------
                xvec[0] = k;
                yvec[0] = j;
                //-------------------------------------------------------
                // Quickly find an adjacent label for use later if needed
                //-------------------------------------------------------
                for( n = 0; n < 4; n++ )
                {
                    int x = xvec[0] + dx4[n];
                    int y = yvec[0] + dy4[n];
                    if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                    {
                        int nindex = y*width + x;
                        if(nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
                    }
                }

                count = 1;
                for( c = 0; c < count; c++ )
                {
                    for( n = 0; n < 4; n++ )
                    {
                        x = xvec[c] + dx4[n];
                        y = yvec[c] + dy4[n];

                        if( (x >= 0 && x < width) && (y >= 0 && y < height) )
                        {
                            int nindex = y*width + x;

                            if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
                            {
                                xvec[count] = x;
                                yvec[count] = y;
                                nlabels[nindex] = label;
                                count++;
                            }
                        }    
                    }
                }
                //-------------------------------------------------------
                // If segment size is less then a limit, assign an
                // adjacent label found before, and decrement label count.
                //-------------------------------------------------------

                if (count <= SUPSZ/maxSUPSZ)
                {
                    for( c = 0; c < count; c++ )
                    {
                        ind = yvec[c]*width+xvec[c];
                        nlabels[ind] = adjlabel;
                    }
                    label--;
                }
                label++;
            }
            oindex++;
        }
    }
    *finalNumberOfLabels = label;

    delete[] xvec;
    delete[] yvec;
}

void uniqueMask(Mat A, vector<int> &values)
{
    int value;
    int it_val = 0;
    bool REP_VAL;

    for (int vi = 0; vi < A.rows; vi++)
    {
        for (int vj = 0; vj < A.cols; vj++)
        {
            value = A.at<int>(vi,vj);
            REP_VAL = false;
            for (int rvi = 0; rvi < it_val; rvi++)
            {
                if (value == values[rvi])
                {
                    REP_VAL = true;
                    break;
                }
            }

            if (!REP_VAL)
            {   
                values.push_back(value);
                it_val++;
            }
        }
    }
}

void uniqueM(Mat A, vector<int> &values)
{
    int value;
    bool REP_VAL;

    for (int vi = 0; vi < A.rows; vi++)
    {
        for (int vj = 0; vj < A.cols; vj++)
        {
            value = A.at<int>(vi,vj);
            REP_VAL = false;
            for (int rvi = 0; rvi < values.size(); rvi++)
            {
                if (value == values[rvi])
                {
                    REP_VAL = true;
                    break;
                }
            }

            if (!REP_VAL)
            {   
                values.push_back(value);
            }
        }
    }
}

void consecLabels(Mat &labels){
    
    Mat newLabels;
    labels.copyTo(newLabels);

    vector<int> numbers;
    uniqueM(newLabels,numbers);
    int newlabel = 0;

    for (int i = 0; i < numbers.size(); i++)
    {
       
        for (int k = 0; k < newLabels.rows; k++)
        {
            for (int j = 0; j < newLabels.cols; j++)
            {
                if (labels.at<int>(k,j) == numbers[i])
                {
                    newLabels.at<int>(k,j) = newlabel;
                }
            }
        }

        newlabel++;        
    }
    newLabels.copyTo(labels);
}


int main(int argc, char** argv )
{
    Mat mean_img;
    bool OUTPUT = false;
    double invwt = 0; 
    bool POSWT = false;
    // WE:
    // string model_name = "/media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/bass-superpixels/c++/bsd_model.bin";
    // IRI:
    // string model_name = "/home/arubio/Documents/Superpixels/bass-superpixels/c++/bsd_model.bin";
    // Model* model = new Model();
    // model->initmodel(model_name);

    if ( argc < 2 )
    {
        printf("\nUsage: bass.out <Image/Image_Folder Path> <Optional arguments>\n\nOptional arguments include:\n\t-s <int>: number of initial seeds\n\t-m <path>: mean image path\n\t-g <double>: geodesic distance weight\n\n");
        return -1;
    }
    
    int numSuperpixels = 50; //default value
    double wtgeo = .5e3; //default value
    double th = 200; //default value
    double compactness = 10; //default value
    double maxSUPSZ = 25; // default value
    int edgeDetectionMethod = 1; // 0: P. Dollar's (default) / 1: OpenCV's Sobel
    string output_dir;  

    for (int iac = 0; iac < argc; iac++)
    {
        if (strcmp(argv[iac],"-s") == 0)
        {
            cout << atoi(argv[iac+1]) << " superpixel seeds" << endl;
            numSuperpixels = atoi(argv[iac+1]);
        }

        if (strcmp(argv[iac],"-g") == 0)
        {
            cout << "Geodesic weight: " << atof(argv[iac+1]) << endl;
            wtgeo = atof(argv[iac+1]);
        }
        
        if (strcmp(argv[iac],"-t") == 0)
        {
            cout << "Background threshold: " << atof(argv[iac+1]) << endl;
            th = atof(argv[iac+1]);
        }

        if (strcmp(argv[iac],"-e") == 0)
        {
            cout << "Euclidean distance weight: " << atof(argv[iac+1]) << endl;
            invwt = atof(argv[iac+1]);
            POSWT = true;
        }

        if (strcmp(argv[iac],"-c") == 0)
        {
            cout << "Compactness: " << atof(argv[iac+1]) << endl;
            compactness = atof(argv[iac+1]);
        }

        if (strcmp(argv[iac],"-d") == 0)
        {
            cout << "Max. superpixel size = SUPSZ/" << atof(argv[iac+1]) << endl;
            maxSUPSZ = atof(argv[iac+1]);
        }

        if (strcmp(argv[iac],"-o") == 0)
        {
            cout << "Output directory: " << argv[iac+1] << endl;
            output_dir = argv[iac+1];
            OUTPUT = true;
        }

        if (strcmp(argv[iac],"-ed") == 0)
        {
            cout << "Edge detection: " << argv[iac+1] << endl;
            edgeDetectionMethod = atoi(argv[iac+1]);
        }
        
        if (strcmp(argv[iac],"-h") == 0)
        {
            printf("\nUsage: bass.out <Image/Image_Folder Path> <Optional arguments>\n\nOptional arguments include:\n\t-s <int>: number of initial seeds\n\t-m <path>: mean image path\n\t-o <path>: output directory\n\t-t <double>: background threshold\n\t-g <double>: geodesic distance weight\n\t-e <double>: euclidean distance weight\n\t-c <double>: compactness\n\t-d <double>: divisor of max. spixel size\n\t-ed <0,1>: edge detection method (0: P.Dollar's, 1: OpenCV's)\n\n");
            return -1;
        }
    }

    vector<Mat> images;
    vector<cv::String> fn;

    bool PATH = false;

    string name = argv[1];
    if (name.find(".jpg") > name.size() && name.find(".png") > name.size() && name.find(".JPEG") > name.size()) {PATH = true;}

    if (PATH)
    {
        string im_path = argv[1];
        
        string im_path1 =im_path + "/*.jpg";
        glob(im_path1, fn, false);

        if (fn.size() == 0)
        {
            string im_path2 =im_path + "/*.png";
            glob(im_path2, fn, false);
        }
        
        if (fn.size() == 0)
        {
            string im_path3 =im_path + "/*.JPEG";
            glob(im_path3, fn, false);
        }    
        
        size_t count = fn.size(); //number of jpg files in images folder
        for (size_t i=0; i<count; i++)
            images.push_back(imread(fn[i]));

        cout << "Reading " << images.size() << " images" << endl;
    }
    else
    {
        cout << "Image " << name << endl;
        fn.push_back(argv[1]);
        images.push_back(imread(argv[1],1));
    }

    int width;
    int height;
    int sz;
    int i, ii;
    int x, y;
    int step;
    int numseeds = 0;
    int k;
    int channels;
    int finalNumberOfLabels;
    int numelements;
    int numChannels;
    int indC, indM;
    int ndel;
    double k_th;
    int im_h;
    int im_w;
    int D;
    int A;
    int B;
    int C;
    int sum1;
    int sum2;
    double shiftFG;
    double shiftBG;
    double shiftFGBG;
    int bgr[] = {0, 0, 204};
    string store_contours;
    string store_mean;
    Mat thEdges;
    Mat intImg;
    Mat image;
    
    cout << "-----------------" << endl;

    for (int imgn = 0; imgn < images.size(); imgn++)
    {
        unsigned char* imgbytes;
        double* ths_P;
        int* rin; int* gin; int* bin;
        int* klabels;
        int* clabels;
        double* lvec; double* avec; double* bvec;
        int* seedIndices;
        int* xseeds;
        int* yseeds;
        double* kseedsx;double* kseedsy;
        double* kseedsl;double* kseedsa;double* kseedsb;

        image = images[imgn];
        
        if ( !image.data )
        {
            printf("No image data \n");
            return -1;
        }

        width = image.cols;  
        height = image.rows;
        sz = width*height;

        thEdges = Mat(height, width, CV_32S, Scalar(0));
        // if(edgeDetectionMethod == 0) // Default: 0 (P. Dollar)
        // {
            // edgeDetection(image, thEdges, model);
        // }
        // else
        // {   
            edgeDetection2(image, thEdges);
        // }

        intImg = Mat(height+1, width+1, CV_32S, Scalar(0));
        integral(thEdges,intImg);

        numseeds = 0;
        channels = image.channels();
        width = image.cols; 
        height = image.rows;
        sz = width*height;
        numelements = image.cols*image.rows*channels;
        numChannels = numelements/sz;
        cout << "width x height x channels: " << width << " x " << height << " x " << numChannels << endl;

        //---------------------------
        // Allocate memory
        //---------------------------
        imgbytes = new unsigned char [numelements];
        rin    = new int [sz];
        gin    = new int [sz];
        bin    = new int [sz];
        lvec    = new double [sz];
        avec    = new double [sz];
        bvec    = new double [sz];
        klabels = new int [sz]; //original labels
        clabels = new int [sz]; //corrected labels after enforcing connectivity
        seedIndices = new int [sz];
        xseeds = new int [sz];
        yseeds = new int [sz];

        for (int li = 0; li < sz; li++)
        {
            rin[li] = 0;
            gin[li] = 0;
            bin[li] = 0;
            lvec[li] = 0;
            avec[li] = 0;
            bvec[li] = 0;
            klabels[li] = 0;
            clabels[li] = 0;
            seedIndices[li] = 0;
            xseeds[li] = 0;
            yseeds[li] = 0; 
        }

        for (int imi = 0; imi < sz*numChannels; imi++)
        {
            // imgbytes_[imi] = 0;
            imgbytes[imi] = 0;
        }

        //---------------------------

        CV_Assert(image.depth() == CV_8U);

        unsigned char* imgbytes_ = image.data;
        
        //------------------------------
        // YUV COLOR SPACE
        //------------------------------
        Mat original_image = image;
        Mat img_out = original_image.clone();
        // cvtColor(original_image, img_out, CV_BGR2YCrCb); // WE
        cvtColor(original_image, img_out, cv::COLOR_BGR2YCrCb); // IRI

        unsigned char* imgbytes_YUV;
        imgbytes_YUV = new unsigned char [numelements];

        for (int imi = 0; imi < sz*numChannels; imi++)
        {
            imgbytes_YUV[imi] = 0;
        }

        unsigned char* imgbytes_YUV_ = img_out.data;

        for(x = 0, ii = 0; x < width; x++)//reading data from column-major MATLAB matrices to row-major C matrices (i.e perform transpose)
        {
            for(y = 0; y < height; y++)
            {
                i = y*width+x;
                ii++;
            }
        }

        bool use_YUV = true;

        if(numelements/sz != 1)// if it is NOT a grayscale image
        {
            for(int ip = 0; ip < height; ip++)
            {
                for (int jp = 0; jp < width; jp++)
                {
                    for(int cp = numChannels-1; cp > -1; cp--)
                    {
                        indC = (width*numChannels)*ip + numChannels*jp + ((numChannels-1)-cp);
                        indM = ip + height*jp + sz*cp;
                        imgbytes[indM] = imgbytes_[indC];       
                        // ------------ YUV color space ------------//
                        imgbytes_YUV[indM] = imgbytes_YUV_[indC];       
                    }
                }
            }
        }

        //---------------------------
        // Perform color conversion
        //---------------------------
        if(numelements/sz == 1)//if it is a grayscale image, copy the values directly into the lab vectors
        {
            for(x = 0, ii = 0; x < width; x++)//reading data from column-major MATLAB matrics to row-major C matrices (i.e perform transpose)
            {
                for(y = 0; y < height; y++)
                {
                    i = y*width+x;
                        lvec[i] = imgbytes[ii];
                        avec[i] = imgbytes[ii];
                        bvec[i] = imgbytes[ii];
                    ii++;
                }
            }
        }
        else //else covert from rgb to lab
        {
            for(x = 0, ii = 0; x < width; x++)//reading data from column-major MATLAB matrics to row-major C matrices (i.e perform transpose)
            {
                for(y = 0; y < height; y++)
                {
                    i = y*width+x;
                    if( use_YUV)
                    {
                        lvec[i] = imgbytes_YUV[ii]; // V channel
                        avec[i] = imgbytes_YUV[ii+sz]; // U channel
                        bvec[i] = 0.2*imgbytes_YUV[ii+sz+sz]; // Y channel
                    }
                    else
                    {
                        rin[i] = imgbytes[ii];
                        gin[i] = imgbytes[ii+sz];
                        bin[i] = imgbytes[ii+sz+sz];
                    }
                    ii++;
                }
            }
            if (!use_YUV)
            {
                rgbtolab(rin,gin,bin,sz,lvec,avec,bvec);
            }
        }


        //---------------------------
        // Find seeds
        //---------------------------
        step = sqrt((double)(sz)/(double)(numSuperpixels))+0.5;

        ndel = 0;
        addDeleteSeeds(xseeds, yseeds, &numseeds, numSuperpixels, intImg, image, &ndel);
        
        k_th = 0.125*th;

        th = k_th*ndel + th;

        cout << "numseeds = " << numseeds << endl;

        kseedsx    = new double [numseeds];
        kseedsy    = new double [numseeds];
        kseedsl    = new double [numseeds];
        kseedsa    = new double [numseeds];
        kseedsb    = new double [numseeds];
        

        for(k = 0; k < numseeds; k++)
        {
            kseedsx[k] = xseeds[k];
            kseedsy[k] = yseeds[k];
            kseedsl[k] = lvec[yseeds[k]*width + xseeds[k]];
            kseedsa[k] = avec[yseeds[k]*width + xseeds[k]];
            kseedsb[k] = bvec[yseeds[k]*width + xseeds[k]];
        }

        //---------------------------
        // Compute superpixels
        //---------------------------
        double pLAB = 0;
        double pXY = 0;
        double pG = 0;

        tic();
        PerformBASS(lvec, avec, bvec, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, width, height, numseeds, klabels, step, compactness, thEdges, wtgeo,th, invwt, POSWT, &pLAB, &pXY, &pG);
        toc();
        
        //---------------------------
        // Enforce connectivity
        //---------------------------
        EnforceSuperpixelConnectivity(klabels,width,height,numSuperpixels,clabels,&finalNumberOfLabels,maxSUPSZ);

        //---------------------------
        // Assign output labels
        //---------------------------
        cout << "Number of labels: " << finalNumberOfLabels << endl;
        Mat labelImg = Mat(height, width, CV_32S, clabels);

        im_h = intImg.rows;
        im_w = intImg.cols;
        D = intImg.at<double>(round(0.9*im_h), round(0.7*im_w));
        A = intImg.at<double>(round(0.1*im_h), round(0.3*im_w));
        B = intImg.at<double>(round(0.1*im_h), round(0.7*im_w));
        C = intImg.at<double>(round(0.9*im_h), round(0.3*im_w));
        sum1 = A + D - B - C; // Inner value
        sum2 = intImg.at<double>(im_h-1,im_w-1) - sum1; // Outer value

        labelImg = labelImg + 1;

        ths_P = new double [2];
        ths_P[0] = 0;
        ths_P[1] = 0;
        shiftFG = 1;
        shiftBG = 2;
        shiftFGBG = 2;

        consecLabels(labelImg);

        int **toDraw;
        toDraw = new int*[height];

        for (int i = 0; i < height; ++i)
        toDraw[i] = new int[width];

        for(x = 0; x < width; x++)//copying data from row-major C matrix to column-major MATLAB matrix (i.e. perform transpose)
        {
            for(y = 0; y < height; y++)
            {
                i = y*width+x;
                toDraw[y][x] = labelImg.at<int>(y,x);  
            }
        }

        Mat contourImage = Draw::contourImage(toDraw, image, bgr);
        Mat labelImage = Draw::labelImage(toDraw, image);
        Mat meanImage = Draw::meanImage(toDraw, image);

        // string img_name_ext = argv[1];
        string img_name_ext = fn[imgn];
        string store;
        const size_t pos = img_name_ext.rfind("/");

        if (OUTPUT)
        {
            store = output_dir + "/" +  img_name_ext.substr(pos+1, img_name_ext.size() - (pos+1) - (img_name_ext.size() - img_name_ext.rfind(".")));
        }
        else
        {
            store = img_name_ext.substr(0,pos) + "/output/" +  img_name_ext.substr(pos+1, img_name_ext.size() - (pos+1) - (img_name_ext.size() - img_name_ext.rfind(".")));
        }
        // string store = "output/" + img_name_ext.substr(0, img_name_ext.size()-4);
 
        /////////////////////////////////////////////////
        // TO SAVE WITH PERCENTAGES OF ENERGY FUNCTION://
        /////////////////////////////////////////////////
        
        stringstream ss_lab;
        ss_lab << round(pLAB);
        string str_lab = ss_lab.str();

        stringstream ss_xy;
        ss_xy << round(pXY);
        string str_xy = ss_xy.str();

        stringstream ss_g;
        ss_g << round(pG);
        string str_g = ss_g.str();
        
        cout << "Saving in " << store << endl;
        store_contours = store + "-lab_" + str_lab  + "xy_" + str_xy + "g_" + str_g + "_contours.png";
        // string store_label = store + "_labels.png";
        store_mean = store + "-lab_" + str_lab  + "xy_" + str_xy + "g_" + str_g + "_mean.png";
        // store_mean = store + "_mean.png";
        imwrite(store_contours, contourImage);
        // imwrite(store_label, labelImage);
        imwrite(store_mean, meanImage);

        string csvFile = store + ".csv";
        Export::CSV(toDraw, image.rows, image.cols, csvFile);

        cout << "-----------------" << endl;
       
        //---------------------------
        // Deallocate memory
        //---------------------------
        delete[] rin;
        delete[] gin;
        delete[] bin;
        delete[] lvec;
        delete[] avec;
        delete[] bvec;
        delete[] kseedsx;
        delete[] kseedsy;
        delete[] kseedsl;
        delete[] kseedsa;
        delete[] kseedsb;
        delete[] klabels;
        delete[] clabels;
        delete[] seedIndices;
        delete[] toDraw;
        delete[] xseeds;
        delete[] yseeds;
        delete[] ths_P;
        // delete[] imgbytes_;
        delete[] imgbytes;

    } // End IMAGES loop
}
