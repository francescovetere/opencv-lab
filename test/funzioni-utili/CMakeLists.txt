cmake_minimum_required(VERSION 2.8)
project(test)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(test
test.cpp
)

target_link_libraries(test
${OpenCV_LIBS}
)

install(TARGETS test RUNTIME DESTINATION bin)
