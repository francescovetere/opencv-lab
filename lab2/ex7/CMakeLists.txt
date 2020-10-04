cmake_minimum_required(VERSION 2.8)
project(ex7)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex7
ex7.cpp
)

target_link_libraries(ex7
${OpenCV_LIBS}
)

install(TARGETS ex7 RUNTIME DESTINATION bin)
