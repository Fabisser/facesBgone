# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.10/site-packages/cmake/data/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.10/site-packages/cmake/data/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/build

# Include any dependencies generated for this target.
include CMakeFiles/hole-filling.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/hole-filling.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/hole-filling.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hole-filling.dir/flags.make

CMakeFiles/hole-filling.dir/src/main.cpp.o: CMakeFiles/hole-filling.dir/flags.make
CMakeFiles/hole-filling.dir/src/main.cpp.o: /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/src/main.cpp
CMakeFiles/hole-filling.dir/src/main.cpp.o: CMakeFiles/hole-filling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hole-filling.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/hole-filling.dir/src/main.cpp.o -MF CMakeFiles/hole-filling.dir/src/main.cpp.o.d -o CMakeFiles/hole-filling.dir/src/main.cpp.o -c /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/src/main.cpp

CMakeFiles/hole-filling.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hole-filling.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/src/main.cpp > CMakeFiles/hole-filling.dir/src/main.cpp.i

CMakeFiles/hole-filling.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hole-filling.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/src/main.cpp -o CMakeFiles/hole-filling.dir/src/main.cpp.s

# Object files for target hole-filling
hole__filling_OBJECTS = \
"CMakeFiles/hole-filling.dir/src/main.cpp.o"

# External object files for target hole-filling
hole__filling_EXTERNAL_OBJECTS =

hole-filling: CMakeFiles/hole-filling.dir/src/main.cpp.o
hole-filling: CMakeFiles/hole-filling.dir/build.make
hole-filling: /usr/local/lib/libgmpxx.dylib
hole-filling: /usr/local/lib/libmpfr.dylib
hole-filling: /usr/local/lib/libgmp.dylib
hole-filling: CMakeFiles/hole-filling.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hole-filling"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hole-filling.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hole-filling.dir/build: hole-filling
.PHONY : CMakeFiles/hole-filling.dir/build

CMakeFiles/hole-filling.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hole-filling.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hole-filling.dir/clean

CMakeFiles/hole-filling.dir/depend:
	cd /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/build /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/build /Users/fabzv/Desktop/Delft/Synthesis-Project/hole-filling/build/CMakeFiles/hole-filling.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hole-filling.dir/depend

