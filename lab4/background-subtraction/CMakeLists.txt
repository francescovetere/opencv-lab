cmake_minimum_required(VERSION 2.8)
project(background-subtraction)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(background-subtraction
background-subtraction.cpp
)

target_link_libraries(background-subtraction
${OpenCV_LIBS}
)

install(TARGETS background-subtraction RUNTIME DESTINATION bin)
