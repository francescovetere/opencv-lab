cmake_minimum_required(VERSION 2.8)
project(lab5_1)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(lab5_1 main.cpp)

target_link_libraries( lab5_1 ${OpenCV_LIBS} )
install(TARGETS lab5_1 RUNTIME DESTINATION bin)
