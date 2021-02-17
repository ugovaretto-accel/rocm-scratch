cmake_minimum_required(VERSION 3.10)

project(first_rocm_cmake_project)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
find_package(hip REQUIRED)

#set(CMAKE_C_COMPILER=clang)
#set(CMAKE_CXX_COMPILER=clang++)
#required to search for libamd_comgr.so.1 and libhsakmt.so.1
link_directories(${ROCM_PATH}/lib64)

set(LIB_TARGET "matrix_transpose")
set(LIB_SOURCE "matrix-transpose.cpp")
add_library(${LIB_TARGET} ${LIB_SOURCE})
target_link_libraries (${LIB_TARGET} PRIVATE hip::device)

set(TRANSPOSE_SOURCES "matrix-transpose-main.cpp")
add_executable (hip_transpose ${TRANSPOSE_SOURCES})
target_link_libraries (hip_transpose matrix_transpose hip::device)

add_executable(threadfence_system threadfence_system.cpp)
#to automatically detect header path and libraries hip::device or hip::host
#is required
target_link_libraries(threadfence_system hip::device)

add_executable(block-matmul block-matrix-mul.cpp)
