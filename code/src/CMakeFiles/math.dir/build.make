# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2

# Include any dependencies generated for this target.
include src/CMakeFiles/math.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/math.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/math.dir/flags.make

src/CMakeFiles/math.dir/math_sse.cpp.o: src/CMakeFiles/math.dir/flags.make
src/CMakeFiles/math.dir/math_sse.cpp.o: src/math_sse.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/math.dir/math_sse.cpp.o"
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/math.dir/math_sse.cpp.o -c /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src/math_sse.cpp

src/CMakeFiles/math.dir/math_sse.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/math.dir/math_sse.cpp.i"
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src/math_sse.cpp > CMakeFiles/math.dir/math_sse.cpp.i

src/CMakeFiles/math.dir/math_sse.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/math.dir/math_sse.cpp.s"
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src/math_sse.cpp -o CMakeFiles/math.dir/math_sse.cpp.s

src/CMakeFiles/math.dir/math_sse.cpp.o.requires:
.PHONY : src/CMakeFiles/math.dir/math_sse.cpp.o.requires

src/CMakeFiles/math.dir/math_sse.cpp.o.provides: src/CMakeFiles/math.dir/math_sse.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/math.dir/build.make src/CMakeFiles/math.dir/math_sse.cpp.o.provides.build
.PHONY : src/CMakeFiles/math.dir/math_sse.cpp.o.provides

src/CMakeFiles/math.dir/math_sse.cpp.o.provides.build: src/CMakeFiles/math.dir/math_sse.cpp.o

# Object files for target math
math_OBJECTS = \
"CMakeFiles/math.dir/math_sse.cpp.o"

# External object files for target math
math_EXTERNAL_OBJECTS =

src/libmath.a: src/CMakeFiles/math.dir/math_sse.cpp.o
src/libmath.a: src/CMakeFiles/math.dir/build.make
src/libmath.a: src/CMakeFiles/math.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libmath.a"
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && $(CMAKE_COMMAND) -P CMakeFiles/math.dir/cmake_clean_target.cmake
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/math.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/math.dir/build: src/libmath.a
.PHONY : src/CMakeFiles/math.dir/build

src/CMakeFiles/math.dir/requires: src/CMakeFiles/math.dir/math_sse.cpp.o.requires
.PHONY : src/CMakeFiles/math.dir/requires

src/CMakeFiles/math.dir/clean:
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && $(CMAKE_COMMAND) -P CMakeFiles/math.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/math.dir/clean

src/CMakeFiles/math.dir/depend:
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2 /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2 /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src/CMakeFiles/math.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/math.dir/depend
