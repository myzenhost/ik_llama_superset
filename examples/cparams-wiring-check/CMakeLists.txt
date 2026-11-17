set(TARGET llama-cparams-wiring-check)
add_executable(${TARGET} cparams-wiring-check.cpp)
install(TARGETS ${TARGET} RUNTIME)
# reads llama_context::get_cparams(), an internal accessor, so it needs src/ on the include path
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(${TARGET} PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
