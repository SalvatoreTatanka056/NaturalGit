cmake_minimum_required(VERSION 3.12)
project(llmasm_c_engine C)

set(CMAKE_C_STANDARD 99)

if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /W3)
    add_definitions(-DLLMASM_EXPORTS)
else()
    add_compile_options(-O3 -mavx2 -mfma -mf16c -Wall -Wextra -fPIC)
    add_definitions(-DLLMASM_EXPORTS)
endif()

add_library(llmasm SHARED llmasm_engine.c llmasm_engine.h)

if(UNIX AND NOT APPLE)
    target_link_libraries(llmasm m)
endif()

set_target_properties(llmasm PROPERTIES
    OUTPUT_NAME "llmasm"
    PREFIX ""
)
