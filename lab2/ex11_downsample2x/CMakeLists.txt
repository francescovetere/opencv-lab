cmake_minimum_required(VERSION 2.8)
project(ex11)

find_package(OpenCV REQUIRED)

add_definitions("-std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

add_executable(ex11
ex11.cpp
)

target_link_libraries(ex11
${OpenCV_LIBS}
)

install(TARGETS ex11 RUNTIME DESTINATION bin)
