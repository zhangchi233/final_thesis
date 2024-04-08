cmake_minimum_required(VERSION 3.0)
project(MyOpenCVProject)

# Set the C++ standard you wish to use (will default to C++98 if not set)
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your source files
add_executable(MyOpenCVProject add_light.cpp) # Replace main.cpp with your source file(s)

# Link OpenCV libraries
target_link_libraries(MyOpenCVProject ${OpenCV_LIBS} pthread)
