cmake_minimum_required(VERSION 2.8)
project(frame-precedente)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(frame-precedente
frame-precedente.cpp
)

target_link_libraries(frame-precedente
${OpenCV_LIBS}
)

install(TARGETS frame-precedente RUNTIME DESTINATION bin)
