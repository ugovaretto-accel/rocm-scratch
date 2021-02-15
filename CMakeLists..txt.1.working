cmake_minimum_required(VERSION 3.8)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCM has been installed")
    elseif(DEFINED ENV{HIP_PATH})
        set(ROCM_PATH "$ENV{HIP_PATH}/.." CACHE PATH "Path to which ROCM has been installed")
    else()
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCM has been installed")
    endif()
endif()

if(NOT DEFINED HCC_PATH)
    if(DEFINED ENV{HCC_PATH})
        set(HCC_PATH $ENV{HCC_PATH} CACHE PATH "Path to which HCC has been installed")
    else()
        set(HCC_PATH "${ROCM_PATH}/hcc" CACHE PATH "Path to which HCC has been installed")
    endif()
    set(HCC_HOME "${HCC_PATH}")
endif()

if(NOT DEFINED HIP_CLANG_PATH)
    if(NOT DEFINED ENV{HIP_CLANG_PATH})
        set(HIP_CLANG_PATH "${ROCM_PATH}/llvm/bin" CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    else()
        set(HIP_CLANG_PATH $ENV{HIP_CLANG_PATH} CACHE PATH "Path to which HIP compatible clang binaries have been installed")
    endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
list(APPEND CMAKE_PREFIX_PATH
    "${HIP_PATH}/lib/cmake"
    "${HIP_PATH}/../lib/cmake" # hopefully catches all extra HIP dependencies
)

project(first_rocm_cmake_project)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()
find_package(hip REQUIRED)

# For ROCm >=3.5, wipe hip-clang specific interface options which are propagated
set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "-fPIC")
set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")


set(MY_SOURCE_FILES "matrix-transpose.cpp")
set(MY_TARGET_NAME "MatrixTranspose")
set(MY_HIPCC_OPTIONS "-fPIC")
set(MY_HCC_OPTIONS )
set(MY_NVCC_OPTIONS )
set(MY_CLANG_OPTIONS )

if (BUILD_SHARED_LIBS)
    set(STATIC_OR_SHARED SHARED)
else()
    set(STATIC_OR_SHARED STATIC)
endif()

link_directories(${ROCM_PATH}/lib64)
set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_library(${MY_TARGET_NAME} ${MY_SOURCE_FILES} HIPCC_OPTIONS "${MY_HIPCC_OPTIONS}" HCC_OPTIONS "${MY_HCC_OPTIONS}" NVCC_OPTIONS "${MY_NVCC_OPTIONS}" CLANG_OPTIONS "${MY_CLANG_OPTIONS}" ${STATIC_OR_SHARED})
set_target_properties(${MY_TARGET_NAME} PROPERTIES LINKER_LANGUAGE HIP)
target_link_libraries (${MY_TARGET_NAME} PRIVATE hip::device)

set (SOURCES "matrix-transpose-main.cpp")
#set (SOURCES "threadfence_system.cpp")
add_executable (hip_cmake_test ${SOURCES})
target_include_directories(hip_cmake_test
    PRIVATE
        $<BUILD_INTERFACE:${HIP_PATH}/include>
        $<BUILD_INTERFACE:${HIP_PATH}/../include>)
target_link_libraries (hip_cmake_test MatrixTranspose hip::device)