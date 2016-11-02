include_directories(../include)
#set(OPENCV_LIBS /usr/local/lib/libopencv_core.a /usr/local/lib/libopencv_highgui.a /usr/local/lib/libopencv_imgcodecs.so)

add_library( edge_detect edgesDetect.cpp edgesNms.cpp model.cpp structureEdge.cpp chnsFunctions.cpp)
set_target_properties(edge_detect PROPERTIES COMPILE_FLAGS "-fPIC") 
target_link_libraries(edge_detect  -Wl,--whole-archive, --no-whole-archive ${OpenCV_LIBS})

add_library( math math_sse.cpp)
target_link_libraries( math )

add_library(edge2box edgeBoxes.cpp)
set_target_properties(edge2box PROPERTIES COMPILE_FLAGS "-fPIC") 
target_link_libraries(edge2box  -Wl,--whole-archive,--no-whole-archive ${OpenCV_LIBS})


