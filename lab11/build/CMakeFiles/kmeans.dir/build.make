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
CMAKE_SOURCE_DIR = /home/francesco/Desktop/VA/lab/lab11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/francesco/Desktop/VA/lab/lab11/build

# Include any dependencies generated for this target.
include CMakeFiles/kmeans.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/kmeans.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/kmeans.dir/flags.make

CMakeFiles/kmeans.dir/kmeans.cpp.o: CMakeFiles/kmeans.dir/flags.make
CMakeFiles/kmeans.dir/kmeans.cpp.o: ../kmeans.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/francesco/Desktop/VA/lab/lab11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/kmeans.dir/kmeans.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/kmeans.dir/kmeans.cpp.o -c /home/francesco/Desktop/VA/lab/lab11/kmeans.cpp

CMakeFiles/kmeans.dir/kmeans.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/kmeans.dir/kmeans.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/francesco/Desktop/VA/lab/lab11/kmeans.cpp > CMakeFiles/kmeans.dir/kmeans.cpp.i

CMakeFiles/kmeans.dir/kmeans.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/kmeans.dir/kmeans.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/francesco/Desktop/VA/lab/lab11/kmeans.cpp -o CMakeFiles/kmeans.dir/kmeans.cpp.s

# Object files for target kmeans
kmeans_OBJECTS = \
"CMakeFiles/kmeans.dir/kmeans.cpp.o"

# External object files for target kmeans
kmeans_EXTERNAL_OBJECTS =

kmeans: CMakeFiles/kmeans.dir/kmeans.cpp.o
kmeans: CMakeFiles/kmeans.dir/build.make
kmeans: /usr/local/lib/libopencv_dnn.so.4.5.0
kmeans: /usr/local/lib/libopencv_gapi.so.4.5.0
kmeans: /usr/local/lib/libopencv_highgui.so.4.5.0
kmeans: /usr/local/lib/libopencv_ml.so.4.5.0
kmeans: /usr/local/lib/libopencv_objdetect.so.4.5.0
kmeans: /usr/local/lib/libopencv_photo.so.4.5.0
kmeans: /usr/local/lib/libopencv_stitching.so.4.5.0
kmeans: /usr/local/lib/libopencv_video.so.4.5.0
kmeans: /usr/local/lib/libopencv_videoio.so.4.5.0
kmeans: /usr/local/lib/libopencv_imgcodecs.so.4.5.0
kmeans: /usr/local/lib/libopencv_calib3d.so.4.5.0
kmeans: /usr/local/lib/libopencv_features2d.so.4.5.0
kmeans: /usr/local/lib/libopencv_flann.so.4.5.0
kmeans: /usr/local/lib/libopencv_imgproc.so.4.5.0
kmeans: /usr/local/lib/libopencv_core.so.4.5.0
kmeans: CMakeFiles/kmeans.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/francesco/Desktop/VA/lab/lab11/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable kmeans"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/kmeans.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/kmeans.dir/build: kmeans

.PHONY : CMakeFiles/kmeans.dir/build

CMakeFiles/kmeans.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/kmeans.dir/cmake_clean.cmake
.PHONY : CMakeFiles/kmeans.dir/clean

CMakeFiles/kmeans.dir/depend:
	cd /home/francesco/Desktop/VA/lab/lab11/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/francesco/Desktop/VA/lab/lab11 /home/francesco/Desktop/VA/lab/lab11 /home/francesco/Desktop/VA/lab/lab11/build /home/francesco/Desktop/VA/lab/lab11/build /home/francesco/Desktop/VA/lab/lab11/build/CMakeFiles/kmeans.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/kmeans.dir/depend

