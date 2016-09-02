// Simple python warpper for LBMoG method
#include <iostream>
#include <opencv2/opencv.hpp>

#include "../package_bgs/jmo/MultiLayerBGS.h"

IBGS *bgs = new MultiLayerBGS;
//MultiLayerBGS bgs;

extern "C" void process(int rows, int cols, unsigned char* imgData,
        unsigned char *fgD, unsigned char *bgD) {
    cv::Mat img(rows, cols, CV_8UC3, (void *) imgData);
    cv::Mat fg(rows, cols, CV_8UC1, fgD);
    cv::Mat bg(rows, cols, CV_8UC1, bgD);
    //bgs.process(img, fg, bg);
    bgs->process(img, fg, bg);
}

extern "C" void setParameters(bool firstTime, bool showOutput, bool saveModel, 
        bool disableDetectMode, bool disableLearning, unsigned char *preload_model)
{    
    MultiLayerBGS::parameters_t param;
    param.firstTime = firstTime;
    param.showOutput = showOutput;
    param.saveModel = saveModel;
    param.disableDetectMode = disableDetectMode;
    param.disableLearning = disableLearning;
    param.preload_model = "";
    // set the parameters
    //bgs.setParameters(param);
    bgs->setParameters(param);
}

