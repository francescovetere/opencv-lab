cmake_minimum_required(VERSION 2.8)
project(ex8)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex8
ex8.cpp
)

target_link_libraries(ex8
${OpenCV_LIBS}
)

install(TARGETS ex8 RUNTIME DESTINATION bin)
