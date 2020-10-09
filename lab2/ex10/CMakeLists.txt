cmake_minimum_required(VERSION 2.8)
project(ex10)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable( ex10
ex10.cpp
)

target_link_libraries(ex10
${OpenCV_LIBS}
)

install(TARGETS ex10 RUNTIME DESTINATION bin)
