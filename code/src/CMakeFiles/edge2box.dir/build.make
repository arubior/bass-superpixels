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
include src/CMakeFiles/edge2box.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/edge2box.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/edge2box.dir/flags.make

src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o: src/CMakeFiles/edge2box.dir/flags.make
src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o: src/edgeBoxes.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o"
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/edge2box.dir/edgeBoxes.cpp.o -c /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src/edgeBoxes.cpp

src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edge2box.dir/edgeBoxes.cpp.i"
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src/edgeBoxes.cpp > CMakeFiles/edge2box.dir/edgeBoxes.cpp.i

src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edge2box.dir/edgeBoxes.cpp.s"
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src/edgeBoxes.cpp -o CMakeFiles/edge2box.dir/edgeBoxes.cpp.s

src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o.requires:
.PHONY : src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o.requires

src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o.provides: src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o.requires
	$(MAKE) -f src/CMakeFiles/edge2box.dir/build.make src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o.provides.build
.PHONY : src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o.provides

src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o.provides.build: src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o

# Object files for target edge2box
edge2box_OBJECTS = \
"CMakeFiles/edge2box.dir/edgeBoxes.cpp.o"

# External object files for target edge2box
edge2box_EXTERNAL_OBJECTS =

src/libedge2box.a: src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o
src/libedge2box.a: src/CMakeFiles/edge2box.dir/build.make
src/libedge2box.a: src/CMakeFiles/edge2box.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libedge2box.a"
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && $(CMAKE_COMMAND) -P CMakeFiles/edge2box.dir/cmake_clean_target.cmake
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/edge2box.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/edge2box.dir/build: src/libedge2box.a
.PHONY : src/CMakeFiles/edge2box.dir/build

src/CMakeFiles/edge2box.dir/requires: src/CMakeFiles/edge2box.dir/edgeBoxes.cpp.o.requires
.PHONY : src/CMakeFiles/edge2box.dir/requires

src/CMakeFiles/edge2box.dir/clean:
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src && $(CMAKE_COMMAND) -P CMakeFiles/edge2box.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/edge2box.dir/clean

src/CMakeFiles/edge2box.dir/depend:
	cd /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2 /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2 /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src /media/wideeyes/5d1d2957-636a-4f8d-b593-941e2c7e0844/wideeyes/Documentos/Superpixels/repo/slic-modified/c++/test2/src/CMakeFiles/edge2box.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/edge2box.dir/depend

