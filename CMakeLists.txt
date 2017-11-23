cmake_minimum_required(VERSION 3.0)
project(Glitter)

set(CMAKE_CXX_STANDARD 11)

include_directories(Glitter/Vendor/glad/include/)

file(GLOB VENDORS_SOURCES Glitter/Vendor/glad/src/glad.c)
file(GLOB PROJECT_SOURCES Glitter/Sources/*.cpp)

add_executable(example ${PROJECT_SOURCES} ${VENDORS_SOURCES})
target_link_libraries(example glfw dl)
