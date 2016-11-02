TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp \
    ../structureEdge.cpp \
    ../SceneFeatures.cpp \
    ../model.cpp \
    ../FastRCNN.cpp \
    ../edgesNms.cpp \
    ../edgesDetect.cpp \
    ../edgeBoxes.cpp \
    ../chnsFunctions.cpp \
    test.cpp \
    ../math_sse.cpp

HEADERS += \
    ../../include/structureEdge.h \
    ../../include/person_craft.h \
    ../../include/model.h \
    ../../include/imagehelper.hpp \
    ../../include/edgesNms.h \
    ../../include/edgesDetect.h \
    ../../include/edgeBoxes.h \
    ../../include/dbFeature.hpp \
    ../../include/config.h \
    ../../include/chnsFunctions.h \
    ../../include/CaffeCls.hpp \
    test.hpp \
    ../../include/math_sse.hpp



INCLUDEPATH += ../../include \
               ../../../caffe-fast-rcnn/include \
               /usr/local/cuda/include \
               ../../../caffe-fast-rcnn/build/src \
               /usr/local/include/opencv2/



LIBS += -Wl,--whole-archive ../../../caffe-fast-rcnn/build/lib/libcaffe.a -Wl,--no-whole-archive
LIBS += -L/usr/local/cuda/lib64/ -lcurand -lcublas -lcudart -lcudnn
LIBS += -lboost_system -lboost_thread -lboost_python -lboost_regex -lboost_iostreams -llmdb  -lpython2.7 -lleveldb
LIBS += -L/usr/local/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lglog -lprotobuf -lhdf5_hl -lhdf5 -lsnappy -lgflags -lboost_serialization
LIBS += -L/opt/OpenBLAS/lib -lopenblas

QMAKE_CXXFLAGS+= -fopenmp -DWITH_PYTHON_LAYER -msse4
QMAKE_LFLAGS +=  -fopenmp
