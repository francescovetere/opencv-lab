cmake_minimum_required(VERSION 2.8)
project(ex4)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex4
ex4.cpp
)

target_link_libraries(ex4
${OpenCV_LIBS}
)

install(TARGETS ex4 RUNTIME DESTINATION bin)
