cmake_minimum_required(VERSION 2.8)
project(ex13)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex13
ex13.cpp
)

target_link_libraries(ex13
${OpenCV_LIBS}
)

install(TARGETS ex13 RUNTIME DESTINATION bin)
