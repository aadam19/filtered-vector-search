cmake_minimum_required(VERSION 3.16)
project(filtered_vector_search_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FAISS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../ACORN")
set(FAISS_LIB "${FAISS_ROOT}/build/faiss/libfaiss.so")

if(NOT EXISTS "${FAISS_LIB}")
    message(FATAL_ERROR "Faiss shared library not found at ${FAISS_LIB}. Build ACORN first.")
endif()

add_library(faiss SHARED IMPORTED)
set_target_properties(faiss PROPERTIES
    IMPORTED_LOCATION "${FAISS_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${FAISS_ROOT}"
)

add_library(fvs_common STATIC
    fvs_common.cpp
)

target_include_directories(fvs_common PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(fvs_common PUBLIC faiss)

add_executable(build_indexes build_indexes.cpp)
target_link_libraries(build_indexes PRIVATE fvs_common)

add_executable(search_filters search_filters.cpp)
target_link_libraries(search_filters PRIVATE fvs_common)
