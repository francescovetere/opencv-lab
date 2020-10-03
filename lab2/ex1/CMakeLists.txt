cmake_minimum_required(VERSION 2.8)
project(ex1)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex1
ex1.cpp
)

target_link_libraries(ex1
${OpenCV_LIBS}
)

install(TARGETS ex1 RUNTIME DESTINATION bin)
