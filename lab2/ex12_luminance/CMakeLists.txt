cmake_minimum_required(VERSION 2.8)
project(ex12)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex12
ex12.cpp
)

target_link_libraries(ex12
${OpenCV_LIBS}
)

install(TARGETS ex12 RUNTIME DESTINATION bin)
