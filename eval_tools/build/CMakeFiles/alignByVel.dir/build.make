# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/akramer/code/radar_ws/src/goggles/eval_tools

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/akramer/code/radar_ws/src/goggles/eval_tools/build

# Include any dependencies generated for this target.
include CMakeFiles/alignByVel.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/alignByVel.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/alignByVel.dir/flags.make

CMakeFiles/alignByVel.dir/alignByVel.cpp.o: CMakeFiles/alignByVel.dir/flags.make
CMakeFiles/alignByVel.dir/alignByVel.cpp.o: ../alignByVel.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/akramer/code/radar_ws/src/goggles/eval_tools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/alignByVel.dir/alignByVel.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alignByVel.dir/alignByVel.cpp.o -c /home/akramer/code/radar_ws/src/goggles/eval_tools/alignByVel.cpp

CMakeFiles/alignByVel.dir/alignByVel.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alignByVel.dir/alignByVel.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/akramer/code/radar_ws/src/goggles/eval_tools/alignByVel.cpp > CMakeFiles/alignByVel.dir/alignByVel.cpp.i

CMakeFiles/alignByVel.dir/alignByVel.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alignByVel.dir/alignByVel.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/akramer/code/radar_ws/src/goggles/eval_tools/alignByVel.cpp -o CMakeFiles/alignByVel.dir/alignByVel.cpp.s

CMakeFiles/alignByVel.dir/alignByVel.cpp.o.requires:

.PHONY : CMakeFiles/alignByVel.dir/alignByVel.cpp.o.requires

CMakeFiles/alignByVel.dir/alignByVel.cpp.o.provides: CMakeFiles/alignByVel.dir/alignByVel.cpp.o.requires
	$(MAKE) -f CMakeFiles/alignByVel.dir/build.make CMakeFiles/alignByVel.dir/alignByVel.cpp.o.provides.build
.PHONY : CMakeFiles/alignByVel.dir/alignByVel.cpp.o.provides

CMakeFiles/alignByVel.dir/alignByVel.cpp.o.provides.build: CMakeFiles/alignByVel.dir/alignByVel.cpp.o


CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o: CMakeFiles/alignByVel.dir/flags.make
CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o: /home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/akramer/code/radar_ws/src/goggles/eval_tools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o -c /home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp > CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.i

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp -o CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.s

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o.requires:

.PHONY : CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o.requires

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o.provides: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o.requires
	$(MAKE) -f CMakeFiles/alignByVel.dir/build.make CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o.provides.build
.PHONY : CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o.provides

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o.provides.build: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o


CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o: CMakeFiles/alignByVel.dir/flags.make
CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o: /home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/akramer/code/radar_ws/src/goggles/eval_tools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o -c /home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp > CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.i

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp -o CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.s

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o.requires:

.PHONY : CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o.requires

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o.provides: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o.requires
	$(MAKE) -f CMakeFiles/alignByVel.dir/build.make CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o.provides.build
.PHONY : CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o.provides

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o.provides.build: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o


CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o: CMakeFiles/alignByVel.dir/flags.make
CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o: /home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/akramer/code/radar_ws/src/goggles/eval_tools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o -c /home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp > CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.i

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp -o CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.s

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o.requires:

.PHONY : CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o.requires

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o.provides: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o.requires
	$(MAKE) -f CMakeFiles/alignByVel.dir/build.make CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o.provides.build
.PHONY : CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o.provides

CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o.provides.build: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o


# Object files for target alignByVel
alignByVel_OBJECTS = \
"CMakeFiles/alignByVel.dir/alignByVel.cpp.o" \
"CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o" \
"CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o" \
"CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o"

# External object files for target alignByVel
alignByVel_EXTERNAL_OBJECTS =

alignByVel: CMakeFiles/alignByVel.dir/alignByVel.cpp.o
alignByVel: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o
alignByVel: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o
alignByVel: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o
alignByVel: CMakeFiles/alignByVel.dir/build.make
alignByVel: /usr/local/lib/libceres.a
alignByVel: /usr/lib/x86_64-linux-gnu/libglog.so
alignByVel: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
alignByVel: /usr/lib/x86_64-linux-gnu/libspqr.so
alignByVel: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
alignByVel: /usr/lib/x86_64-linux-gnu/libtbb.so
alignByVel: /usr/lib/x86_64-linux-gnu/libcholmod.so
alignByVel: /usr/lib/x86_64-linux-gnu/libccolamd.so
alignByVel: /usr/lib/x86_64-linux-gnu/libcamd.so
alignByVel: /usr/lib/x86_64-linux-gnu/libcolamd.so
alignByVel: /usr/lib/x86_64-linux-gnu/libamd.so
alignByVel: /usr/lib/x86_64-linux-gnu/liblapack.so
alignByVel: /usr/lib/x86_64-linux-gnu/libf77blas.so
alignByVel: /usr/lib/x86_64-linux-gnu/libatlas.so
alignByVel: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
alignByVel: /usr/lib/x86_64-linux-gnu/librt.so
alignByVel: /usr/lib/x86_64-linux-gnu/libcxsparse.so
alignByVel: /usr/lib/x86_64-linux-gnu/libtbbmalloc.so
alignByVel: /usr/lib/x86_64-linux-gnu/libtbb.so
alignByVel: /usr/lib/x86_64-linux-gnu/libcholmod.so
alignByVel: /usr/lib/x86_64-linux-gnu/libccolamd.so
alignByVel: /usr/lib/x86_64-linux-gnu/libcamd.so
alignByVel: /usr/lib/x86_64-linux-gnu/libcolamd.so
alignByVel: /usr/lib/x86_64-linux-gnu/libamd.so
alignByVel: /usr/lib/x86_64-linux-gnu/liblapack.so
alignByVel: /usr/lib/x86_64-linux-gnu/libf77blas.so
alignByVel: /usr/lib/x86_64-linux-gnu/libatlas.so
alignByVel: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
alignByVel: /usr/lib/x86_64-linux-gnu/librt.so
alignByVel: /usr/lib/x86_64-linux-gnu/libcxsparse.so
alignByVel: CMakeFiles/alignByVel.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/akramer/code/radar_ws/src/goggles/eval_tools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable alignByVel"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/alignByVel.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/alignByVel.dir/build: alignByVel

.PHONY : CMakeFiles/alignByVel.dir/build

CMakeFiles/alignByVel.dir/requires: CMakeFiles/alignByVel.dir/alignByVel.cpp.o.requires
CMakeFiles/alignByVel.dir/requires: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalImuVelocityCostFunction.cpp.o.requires
CMakeFiles/alignByVel.dir/requires: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/GlobalVelocityMeasCostFunction.cpp.o.requires
CMakeFiles/alignByVel.dir/requires: CMakeFiles/alignByVel.dir/home/akramer/code/radar_ws/src/goggles/src/QuaternionParameterization.cpp.o.requires

.PHONY : CMakeFiles/alignByVel.dir/requires

CMakeFiles/alignByVel.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/alignByVel.dir/cmake_clean.cmake
.PHONY : CMakeFiles/alignByVel.dir/clean

CMakeFiles/alignByVel.dir/depend:
	cd /home/akramer/code/radar_ws/src/goggles/eval_tools/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/akramer/code/radar_ws/src/goggles/eval_tools /home/akramer/code/radar_ws/src/goggles/eval_tools /home/akramer/code/radar_ws/src/goggles/eval_tools/build /home/akramer/code/radar_ws/src/goggles/eval_tools/build /home/akramer/code/radar_ws/src/goggles/eval_tools/build/CMakeFiles/alignByVel.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/alignByVel.dir/depend

