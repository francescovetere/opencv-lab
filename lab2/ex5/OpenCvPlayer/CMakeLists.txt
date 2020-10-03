cmake_minimum_required(VERSION 2.8.11)
project(OpenCvPlayer)

set(PACKAGE_VENDOR "Vislab")

add_compile_options(-std=c++11 -Wall )

option(USE_QT5  "Use library QT v5.x" ON )

find_package( OpenCV REQUIRED )
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5Network)
find_package(Qt5OpenGL)
find_package(Qt5PrintSupport)
set(QT_LIBRARIES Qt5::Core Qt5::Widgets Qt5::Network Qt5::OpenGL Qt5::PrintSupport)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

set(CMAKE_AUTOMOC ON)

include_directories(
  ${QT_INCLUDES} 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(Engine)

set(OpenCvPlayer_SRCS
      Engine/QtEngine.cpp
      Dummy.cpp 
      main.cpp)

add_executable(OpenCvPlayer ${OpenCvPlayer_SRCS})

target_link_libraries(OpenCvPlayer 
  ${QT_LIBRARIES}
  ${OpenCV_LIBS}
)


install(TARGETS OpenCvPlayer RUNTIME DESTINATION bin)
