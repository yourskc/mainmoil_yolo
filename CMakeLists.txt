cmake_minimum_required(VERSION 2.8)
project( mainmoil_yolo )
find_package( OpenCV 4.0 REQUIRED )
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)
add_executable( mainmoil_yolo ${viso} )
#add_library( moildev STATIC IMPORTED)
#SET_PROPERTY ( TARGET moildev PROPERTY IMPORTED_LOCATION ../moildev.a )
#target_link_libraries( mainmoil_yolo moildev ${OpenCV_LIBS} )
target_link_libraries( mainmoil_yolo -lmoildev ${OpenCV_LIBS} )
