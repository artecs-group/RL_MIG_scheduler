# Set the CUDA Toolkit root directory
set(CUDA_PATH "/usr/local/cuda-12.3" CACHE PATH "CUDA Toolkit root directory")

cmake_minimum_required(VERSION 3.10)
project(MIGScheduler)

find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(STATUS "Directory compile path unknown. Redirect to 'build'.")
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/build")
    execute_process(COMMAND chmod 777 "${CMAKE_SOURCE_DIR}/build")
    # Reinicia CMake con el nuevo directorio de compilación
    execute_process(
        COMMAND ${CMAKE_COMMAND} -B "${CMAKE_SOURCE_DIR}/build" -S "${CMAKE_SOURCE_DIR}"
    )
endif()

# Add the header files
include_directories(include)

set(CUDA_INCLUDE_PATH "${CUDA_PATH}/include")
include_directories(${CUDA_INCLUDE_PATH})

set(NVML_LIB_PATH "${CUDA_PATH}/targets/x86_64-linux/lib/stubs")
link_directories(${NVML_LIB_PATH})

# Add the source files
file(GLOB SOURCES "src/*.cpp")

# Add the executable in normal mode
add_executable(mig_scheduler.exe ${SOURCES})
target_link_libraries(mig_scheduler.exe ${CUDA_LIBRARIES} nvidia-ml)

# Add the executable in debug mode
add_executable(mig_scheduler_debug.exe ${SOURCES})
target_link_libraries(mig_scheduler_debug.exe ${CUDA_LIBRARIES} nvidia-ml)
target_compile_definitions(mig_scheduler_debug.exe PRIVATE DEBUG_MODE)