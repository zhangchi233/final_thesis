cmake_minimum_required(VERSION 3.10)
project(ShadowEffect)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(ShadowEffect main.cpp)

# Link libraries
target_link_libraries(ShadowEffect ${OpenCV_LIBS} pthread)
