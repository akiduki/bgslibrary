import numpy as np
import ctypes as C
import cv2

import pdb

import copy
import sys
import platform
import inspect

import xml.etree.ElementTree as ET

# from mosaic_helper import switch, mergeImages
# import sys.float_info as sysfloat

# libmog = C.cdll.LoadLibrary('./liblbmog.dylib') # uses "../package_bgs/lb/LBMixtureOfGaussians.h"
# libmog = C.cdll.LoadLibrary('./libmlayerBGS.dylib') # uses "../package_bgs/jmo/MultiLayerBGS.h"
# libmog = C.cdll.LoadLibrary('./libwrenBGS.dylib') # uses "../package_bgs/dp/DPWrenGABGS.h"

class BackGroundModel(object):
    
    def __init__(self, method='mlayerBGS'):

        if platform.system() == 'Linux':
            method_name = './lib' + method + '.so'
        elif platform.system() == 'Darwin':
            method_name = './lib' + method + '.dylib'
        else:
            method_name = './lib' + method + '.dll'

        try:
            # pre-requistie: lib must exist
            self.bgMethod = C.cdll.LoadLibrary(method_name)
        except IOError as e:
            print e

    # to be adaptive changed
    def setParameters(self, param):
        self.bgMethod.setParameters(param['firstTime'], param['showOutput'], 
                                    param['saveModel'], param['disableDetectMode'],
                                    param['disableLearning'], param['preload_model'])

    def modifyParameters(self, XMLfile, param):
        # support parameter modification through XML config files
        tree = ET.parse(XMLfile)
        root = tree.getroot()

        # iterate param dictionary to modify the configuration
        for key, value in param.iteritems():
            elem = tree.find(key)
            if elem:
                elem.text = str(value)

        tree.write(XMLfile)

    def process_3ch(self, img):
        (rows, cols) = (img.shape[0], img.shape[1])
        res = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))
        bg = copy.copy(res)
        # apply background modeling
        self.bgMethod.process(rows, cols,
                              img.ctypes.data_as(C.POINTER(C.c_ubyte)),
                              res.ctypes.data_as(C.POINTER(C.c_ubyte)),
                              bg.ctypes.data_as(C.POINTER(C.c_ubyte)))
        return res, bg

    def process(self, img):
        # 1-channel mode
        (rows, cols) = (img.shape[0], img.shape[1])
        res = np.zeros(dtype=np.uint8, shape=(rows, cols, 1))
        bg = copy.copy(res)

        self.bgMethod.process(rows, cols, 
                       img.ctypes.data_as(C.POINTER(C.c_ubyte)),
                       res.ctypes.data_as(C.POINTER(C.c_ubyte)),
                       bg.ctypes.data_as(C.POINTER(C.c_ubyte)))
        return res, bg
    
if __name__ == '__main__':
    videofile = '/Users/yuanyi.xue/Desktop/Baseball 8K tests/Archive/cropped_homerun_x264.mkv'
    videoout = videofile[:-4]+'_bg_libmog.mkv'
    CODEC_TYPE = cv2.cv.FOURCC('F','M','P','4')
    FIRST_FRAME = True
    c = cv2.VideoCapture(videofile)

    while 1:
        _, f = c.read()
        if f is None:
            break
        f_resz = cv2.resize(f, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        # cv2.imshow('f', f_resz)
        # pdb.set_trace()
        fg, bg = getfgbg_1c(f_resz) # one channel foreground return
        fg = fg.astype(np.float)
        fg *= 1.0/(fg.max()+sys.float_info.epsilon) # add espilon to avoid ZeroDivisionError
        if len(fg.shape)==2:
            fg_expand = fg[:,:,np.newaxis]
            fg_show = fg_expand.astype(np.uint8)*f_resz
        else:
            fg_show =  fg.astype(np.uint8)*f_resz
        imgList = [f_resz, fg_show]
        mergedImg = mergeImages(imgList)
        
        if FIRST_FRAME:
            vid = cv2.VideoWriter(videoout, CODEC_TYPE, 60, mergedImg.shape[1:None:-1], 1)
            FIRST_FRAME = False
            
        vid.write(mergedImg)
        cv2.imshow('fg', mergedImg)
        # cv2.imshow('fg', fg)
        if cv2.waitKey(1) == 27:
            exit(0)

    vid.release()
