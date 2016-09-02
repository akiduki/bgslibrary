cmake_minimum_required(VERSION 2.8)

project(BGSPythonWrapper)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

find_package(OpenCV REQUIRED)

# Adding all wrappers
file(GLOB source_mBGS ../example_macosx/w_mlayerBGS.cpp)
file(GLOB source_adaptBGS ../example_macosx/w_adapBGS.cpp)
file(GLOB source_eigenBGS ../example_macosx/w_DPEigenbackgroundBGS.cpp)
file(GLOB source_wrenGABAGS ../example_macosx/w_DPWrenGABGAS.cpp)
file(GLOB source_adaptSOM ../example_macosx/w_LBAdaptiveSOM.cpp)
file(GLOB source_LBMoG ../example_macosx/w_LBMoG.cpp)
file(GLOB_RECURSE bgs_src ../package_bgs/*.cpp ../package_bgs/*.c)
file(GLOB_RECURSE bgs_include ../package_bgs/*.h)

include_directories(${CMAKE_SOURCE_DIR})

add_library(bgs SHARED ${bgs_src})
target_link_libraries(bgs ${OpenCV_LIBS})
set_property(TARGET bgs PROPERTY PUBLIC_HEADER ${bgs_include})

# add SHARED to compile as dylib
add_library(mlayerBGS SHARED ${source_mBGS})
target_link_libraries(mlayerBGS ${OpenCV_LIBS} bgs)
add_library(adaptBGS SHARED ${source_adaptBGS})
target_link_libraries(adaptBGS ${OpenCV_LIBS} bgs)
add_library(wrenBGS SHARED ${source_wrenGABAGS})
target_link_libraries(wrenBGS ${OpenCV_LIBS} bgs)
add_library(lbMoG SHARED ${source_LBMoG})
target_link_libraries(lbMoG ${OpenCV_LIBS} bgs)
add_library(adaptSOM SHARED ${source_adaptSOM})
target_link_libraries(adaptSOM ${OpenCV_LIBS} bgs)
add_library(eigenBGS SHARED ${source_eigenBGS})
target_link_libraries(eigenBGS ${OpenCV_LIBS} bgs)
