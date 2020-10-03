cmake_minimum_required(VERSION 2.8)
project(ex3)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex3
ex3.cpp
)

target_link_libraries(ex3
${OpenCV_LIBS}
)

install(TARGETS ex3 RUNTIME DESTINATION bin)
