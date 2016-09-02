// Simple python warpper for adaptiveBGS method
#include <iostream>
#include <opencv2/opencv.hpp>

#include "../package_bgs/AdaptiveBackgroundLearning.h"

IBGS *bgs = new AdaptiveBackgroundLearning;
//LBMixtureOfGaussians bgs;

extern "C" void process(int rows, int cols, unsigned char* imgData,
        unsigned char *fgD, unsigned char *bgD) {
    cv::Mat img(rows, cols, CV_8UC3, (void *) imgData);
    cv::Mat fg(rows, cols, CV_8UC1, fgD);
    cv::Mat bg(rows, cols, CV_8UC1, bgD);
    //bgs.process(img, fg, bg);
    bgs->process(img, fg, bg);
}

extern "C" void setParameters(bool firstTime, bool showOutput) 
{    
    AdaptiveBackgroundLearning::parameters_t param;
    param.firstTime = firstTime;
    param.showForeground = showOutput;
    param.showBackground = showOutput;
    // set the parameters
    bgs->setParameters(param);
}
