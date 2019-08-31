cmake_minimum_required(VERSION 2.8)

project( bass )

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 4.7)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -msse -msse2 -msse3 -MMD -MP  -pthread -fPIC -DNDEBUG -O2 -Wno-sign-compare -fopenmp")
    else ()
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wno-sign-compare -pedantic")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -std=c++11 -msse -msse2 -msse3 -MMD -MP  -pthread -fPIC -DNDEBUG -O2 -DWITH_PYTHON_LAYER  -Wno-sign-compare -fopenmp")
    endif ()
else ()
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse -msse2 -msse3")
endif ()

find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED COMPONENTS filesystem system)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS}/boost)
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

include_directories(./include/)

option(USE_GPU "Use GPU instead of CPU" 0)

if (USE_GPU)
    include_directories(/usr/local/cuda/include)
endif()
include_directories(/usr/include/python2.7)
include_directories(/usr/lib/python2.7/dist-packages/numpy/core/include)
include_directories(/opt/OpenBLAS/include)


LINK_DIRECTORIES(/opt/OpenBLAS/lib)
add_subdirectory(src)

set(BUILD_SHARED_LIBS OFF)
# if(BUILD_SHARED_LIBS)
#   set(Caffe_LINK caffe)
# else()
#   if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#     set(Caffe_LINK -Wl,-force_load caffe)
#   elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#     set(Caffe_LINK -Wl,--whole-archive ../../caffe-fast-rcnn/build/libcaffe.a )
#   endif()
# endif()

message(STATUS "${Caffe_LINK}")
message(STATUS "${BUILD_SHARED_LIBS}")

if (USE_GPU)
else()
    add_compile_options(-DCPU_ONLY)
endif()

include_directories(include)

include_directories( ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

# add_library( FastRCNN src/FastRCNN.cpp src/SceneFeatures.cpp)

add_executable( bass bass.cpp Tools.cpp )

# set_target_properties(FastRCNN PROPERTIES COMPILE_FLAGS "-fPIC")
set_target_properties(bass PROPERTIES COMPILE_FLAGS "-fPIC")
if (USE_GPU)
    # target_link_libraries( bass FastRCNN ${Boost_LIBRARIES} ${Caffe_LINK} ${OpenCV_LIBS} edge2box edge_detect math boost_thread boost_system ${TBB_LIBRARIES} openblas lapack blas cudart cublas curand glog protobuf snappy lmdb hdf5_hl hdf5)
    target_link_libraries( bass ${Boost_LIBRARIES} ${Caffe_LINK} ${OpenCV_LIBS} edge2box edge_detect math boost_thread boost_system ${TBB_LIBRARIES} openblas lapack blas cudart cublas curand glog protobuf snappy lmdb hdf5_hl hdf5)
else()
    # target_link_libraries( bass FastRCNN ${Boost_LIBRARIES} ${Caffe_LINK} ${OpenCV_LIBS} edge2box edge_detect math boost_thread boost_system ${TBB_LIBRARIES} openblas lapack glog protobuf snappy lmdb hdf5_hl hdf5 opencv_core)
    target_link_libraries( bass ${Boost_LIBRARIES} ${Caffe_LINK} ${OpenCV_LIBS} edge2box edge_detect math boost_thread boost_system ${TBB_LIBRARIES} openblas lapack glog protobuf snappy lmdb hdf5_hl hdf5 opencv_core)
endif()
