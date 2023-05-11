cmake_minimum_required(VERSION 3.10)
PROJECT(nanodet VERSION 1.0.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

find_package(OpenCV REQUIRED)
set(OpenVINO_DIR /usr/lib/x86_64-linux-gnu/cmake/openvino2022.3.0)
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)

include_directories(
        ${InferenceEngine_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
)

list(APPEND ALL_LIBS
        ${InferenceEngine_LIBRARIES}
        ${NGRAPH_LIBRARIES}
        ${OpenCV_LIBS}
        )

add_library(${PROJECT_NAME} SHARED Nanodet.cpp)
target_link_libraries(${PROJECT_NAME} ${ALL_LIBS})

add_executable(pro main.cpp)
target_link_libraries(
        pro
        ${PROJECT_NAME}
        ${ALL_LIBS}
)
