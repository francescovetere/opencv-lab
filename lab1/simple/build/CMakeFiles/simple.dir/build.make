# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/francesco/Desktop/VA/lab/lab1/simple

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/francesco/Desktop/VA/lab/lab1/simple/build

# Include any dependencies generated for this target.
include CMakeFiles/simple.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/simple.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/simple.dir/flags.make

CMakeFiles/simple.dir/simple.cpp.o: CMakeFiles/simple.dir/flags.make
CMakeFiles/simple.dir/simple.cpp.o: ../simple.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesco/Desktop/VA/lab/lab1/simple/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/simple.dir/simple.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/simple.dir/simple.cpp.o -c /home/francesco/Desktop/VA/lab/lab1/simple/simple.cpp

CMakeFiles/simple.dir/simple.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/simple.dir/simple.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesco/Desktop/VA/lab/lab1/simple/simple.cpp > CMakeFiles/simple.dir/simple.cpp.i

CMakeFiles/simple.dir/simple.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/simple.dir/simple.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesco/Desktop/VA/lab/lab1/simple/simple.cpp -o CMakeFiles/simple.dir/simple.cpp.s

# Object files for target simple
simple_OBJECTS = \
"CMakeFiles/simple.dir/simple.cpp.o"

# External object files for target simple
simple_EXTERNAL_OBJECTS =

simple: CMakeFiles/simple.dir/simple.cpp.o
simple: CMakeFiles/simple.dir/build.make
simple: /usr/local/lib/libopencv_dnn.so.4.5.0
simple: /usr/local/lib/libopencv_gapi.so.4.5.0
simple: /usr/local/lib/libopencv_highgui.so.4.5.0
simple: /usr/local/lib/libopencv_ml.so.4.5.0
simple: /usr/local/lib/libopencv_objdetect.so.4.5.0
simple: /usr/local/lib/libopencv_photo.so.4.5.0
simple: /usr/local/lib/libopencv_stitching.so.4.5.0
simple: /usr/local/lib/libopencv_video.so.4.5.0
simple: /usr/local/lib/libopencv_videoio.so.4.5.0
simple: /usr/local/lib/libopencv_imgcodecs.so.4.5.0
simple: /usr/local/lib/libopencv_calib3d.so.4.5.0
simple: /usr/local/lib/libopencv_features2d.so.4.5.0
simple: /usr/local/lib/libopencv_flann.so.4.5.0
simple: /usr/local/lib/libopencv_imgproc.so.4.5.0
simple: /usr/local/lib/libopencv_core.so.4.5.0
simple: CMakeFiles/simple.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/francesco/Desktop/VA/lab/lab1/simple/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable simple"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/simple.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/simple.dir/build: simple

.PHONY : CMakeFiles/simple.dir/build

CMakeFiles/simple.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/simple.dir/cmake_clean.cmake
.PHONY : CMakeFiles/simple.dir/clean

CMakeFiles/simple.dir/depend:
	cd /home/francesco/Desktop/VA/lab/lab1/simple/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/francesco/Desktop/VA/lab/lab1/simple /home/francesco/Desktop/VA/lab/lab1/simple /home/francesco/Desktop/VA/lab/lab1/simple/build /home/francesco/Desktop/VA/lab/lab1/simple/build /home/francesco/Desktop/VA/lab/lab1/simple/build/CMakeFiles/simple.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/simple.dir/depend

