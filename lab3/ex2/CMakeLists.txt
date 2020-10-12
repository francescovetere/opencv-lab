cmake_minimum_required(VERSION 2.8)
project(ex2)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex2
ex2.cpp
)

target_link_libraries(ex2
${OpenCV_LIBS}
)

install(TARGETS ex2 RUNTIME DESTINATION bin)
