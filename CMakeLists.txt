cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static libs are linked into the qwen3tts shared lib (C API), so they must be PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64" AND NOT APPLE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=oryon-1+fp16+fp16fml+dotprod+i8mm+bf16")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
elseif(MSVC)
    # Parallel compilation within each target (significant speedup for large files)
    add_compile_options(/MP)
    # Fast math for Release builds
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /fp:fast")
endif()

# Clang-cl on Windows (Ninja build): enable native CPU optimizations (AVX2/AVX512)
# clang-cl identifies as "Clang" but is invoked via MSVC-compatible flags on Windows
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")
    message(STATUS "Clang-cl: enabling -march=native for AVX2/AVX512")
endif()

# Suppress MSVC/clang-cl "unsafe function" deprecation warnings for standard
# C functions (fopen, sscanf, strncpy, getenv, etc.).
# This matches the approach used by llama.cpp and whisper.cpp.
# These are Windows-only warnings with no cross-platform safety value.
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# KV cache precision option
# F32 KV cache gives bit-exact non-streaming output matching Python float32 inference.
# F16 (default) uses half the memory but may differ by 1 frame at EOS boundary.
option(QWEN3_TTS_KV_F32 "Use F32 KV cache for bit-exact non-streaming output (2× RAM)" OFF)
if(QWEN3_TTS_KV_F32)
    add_compile_definitions(QWEN3_TTS_KV_F32)
    message(STATUS "KV cache: F32 (bit-exact mode, 2× RAM)")
else()
    message(STATUS "KV cache: F16 (default, half RAM)")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
if(NOT DEFINED GGML_BUILD_DIR)
    set(GGML_BUILD_DIR "${GGML_DIR}/build")
endif()

# GGML link search dirs: flat (Ninja/Make) and config-suffixed (MSVC multi-config)
set(GGML_LINK_DIRS
    ${GGML_BUILD_DIR}/bin
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/Release
    ${GGML_BUILD_DIR}/src/Debug
    ${GGML_BUILD_DIR}/src/ggml-metal
    ${GGML_BUILD_DIR}/src/ggml-vulkan
    ${GGML_BUILD_DIR}/src/ggml-opencl
    ${GGML_BUILD_DIR}/src/ggml-hexagon
    ${GGML_BUILD_DIR}/src/ggml-cuda
)

# Detect Metal backend availability
if(APPLE AND EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib")
    set(GGML_HAS_METAL ON)
    message(STATUS "GGML Metal backend found — GPU acceleration enabled (Apple)")
else()
    set(GGML_HAS_METAL OFF)
endif()

# Detect CUDA backend availability
if(EXISTS "${GGML_BUILD_DIR}/src/ggml-cuda")
    file(GLOB GGML_CUDA_LIBS
        "${GGML_BUILD_DIR}/bin/*ggml-cuda*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/*ggml-cuda*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/ggml-cuda/*ggml-cuda*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/Release/*ggml-cuda*${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(GGML_CUDA_LIBS)
        set(GGML_HAS_CUDA ON)
        message(STATUS "GGML CUDA backend found — GPU acceleration enabled (NVIDIA)")
    else()
        set(GGML_HAS_CUDA OFF)
    endif()
else()
    set(GGML_HAS_CUDA OFF)
endif()

# Detect Vulkan backend availability
if(EXISTS "${GGML_BUILD_DIR}/src/ggml-vulkan")
    file(GLOB GGML_VULKAN_LIBS
        "${GGML_BUILD_DIR}/bin/*ggml-vulkan*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/*ggml-vulkan*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/ggml-vulkan/*ggml-vulkan*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/Release/*ggml-vulkan*${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(GGML_VULKAN_LIBS)
        set(GGML_HAS_VULKAN ON)
        message(STATUS "GGML Vulkan backend found — GPU acceleration enabled (cross-platform)")
    else()
        set(GGML_HAS_VULKAN OFF)
    endif()
else()
    set(GGML_HAS_VULKAN OFF)
endif()

# Detect OpenCL backend availability
if(EXISTS "${GGML_BUILD_DIR}/src/ggml-opencl")
    file(GLOB GGML_OPENCL_LIBS
        "${GGML_BUILD_DIR}/bin/*ggml-opencl*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/*ggml-opencl*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/ggml-opencl/*ggml-opencl*${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${GGML_BUILD_DIR}/src/Release/*ggml-opencl*${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(GGML_OPENCL_LIBS)
        set(GGML_HAS_OPENCL ON)
        message(STATUS "GGML OpenCL backend found — GPU acceleration enabled (Adreno/OpenCL)")
    else()
        set(GGML_HAS_OPENCL OFF)
    endif()
else()
    set(GGML_HAS_OPENCL OFF)
endif()

if(NOT GGML_HAS_METAL AND NOT GGML_HAS_CUDA AND NOT GGML_HAS_VULKAN AND NOT GGML_HAS_OPENCL)
    message(STATUS "No GPU backend detected — running on CPU only")
    message(STATUS "  For GPU acceleration:")
    message(STATUS "    macOS:   cmake -S ggml -B ggml/build -DGGML_METAL=ON")
    message(STATUS "    Windows: cmake -S ggml -B ggml/build -DGGML_CUDA=ON  (needs CUDA toolkit)")
    message(STATUS "    Any GPU: cmake -S ggml -B ggml/build -DGGML_VULKAN=ON (needs Vulkan SDK)")
endif()

# gguf_loader utility — shared by all components, compiled once
add_library(gguf_loader STATIC
    src/gguf_loader.cpp
)
target_include_directories(gguf_loader PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(gguf_loader PUBLIC ${GGML_LINK_DIRS})
target_link_libraries(gguf_loader PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(gguf_loader PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()
if(GGML_HAS_VULKAN)
    target_link_libraries(gguf_loader PUBLIC ggml-vulkan)
endif()
if(GGML_HAS_OPENCL)
    target_link_libraries(gguf_loader PUBLIC ggml-opencl)
endif()
if(GGML_HAS_CUDA)
    target_link_libraries(gguf_loader PUBLIC ggml-cuda)
endif()

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(text_tokenizer PUBLIC
    gguf_loader
)

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(tts_transformer PUBLIC
    gguf_loader
)
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(audio_tokenizer_encoder PUBLIC
    gguf_loader
)

# Mimi audio encoder library (CPU-based, for ICL voice cloning)
add_library(mimi_encoder STATIC
    src/mimi_encoder.cpp
)
target_include_directories(mimi_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(mimi_encoder PUBLIC
    gguf_loader
)

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(audio_tokenizer_decoder PUBLIC
    gguf_loader
)

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    mimi_encoder
    audio_tokenizer_decoder
    Threads::Threads
)
if(WIN32)
    target_link_libraries(qwen3_tts PUBLIC psapi)
endif()

# Shared library with C API (for Nim FFI)
add_library(qwen3tts_shared SHARED
    src/qwen3tts_c_api.cpp
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_shared PRIVATE
    qwen3_tts
    audio_player
)

# miniaudio-backed blocking playback (CLI --speak)
add_library(audio_player STATIC
    src/audio_player.cpp
)
target_include_directories(audio_player PRIVATE ${CMAKE_SOURCE_DIR}/lib/miniaudio)
if(APPLE)
    target_link_libraries(audio_player PUBLIC
        "-framework CoreFoundation"
        "-framework CoreAudio"
        "-framework AudioToolbox"
        "-framework AudioUnit"
    )
elseif(WIN32)
    target_link_libraries(audio_player PUBLIC ole32)
endif()

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_include_directories(qwen3-tts-cli PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
    audio_player
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# Test executable for Mimi encoder (validates ICL reference audio encoding)
add_executable(test_mimi_encoder
    tests/test_mimi_encoder.cpp
)
target_link_libraries(test_mimi_encoder PRIVATE
    mimi_encoder
    Threads::Threads
)

# Test: VQ codebook encoding/decoding round-trip
add_executable(test_codebook
    tests/test_codebook.cpp
)
target_link_libraries(test_codebook PRIVATE
    audio_tokenizer_decoder
    gguf_loader
    Threads::Threads
)

# Test: VQ-only encode/decode
add_executable(test_vq_only
    tests/test_vq_only.cpp
)
target_link_libraries(test_vq_only PRIVATE
    audio_tokenizer_decoder
    gguf_loader
    Threads::Threads
)

# Test executable for batch inference validation
add_executable(test_batch
    tests/test_batch.cpp
)
target_link_libraries(test_batch PRIVATE
    tts_transformer
    text_tokenizer
    gguf_loader
    Threads::Threads
)

# Batch throughput benchmark
add_executable(bench_batch
    tests/bench_batch.cpp
)
target_link_libraries(bench_batch PRIVATE
    tts_transformer
    text_tokenizer
    gguf_loader
    Threads::Threads
)

# On Windows, copy GGML DLLs to the output directory so executables can find them at runtime.
# GGML is built separately (ggml/build) and its DLLs are not automatically staged by CMake.
# Copies core DLLs + any GPU backend DLLs that were built (CUDA, Vulkan).
if(WIN32)
    set(GGML_DLL_SEARCH_DIRS
        ${GGML_BUILD_DIR}/bin
        ${GGML_BUILD_DIR}/src/Release
        ${GGML_BUILD_DIR}/src/Debug
        ${GGML_BUILD_DIR}/src
    )
    # Core DLLs — always required
    set(GGML_CORE_DLLS ggml.dll ggml-base.dll ggml-cpu.dll)
    # GPU backend DLLs — copied only if built
    set(GGML_GPU_DLLS ggml-cuda.dll ggml-cuda-f16.dll ggml-vulkan.dll ggml-sycl.dll)
    foreach(_dll ${GGML_CORE_DLLS} ${GGML_GPU_DLLS})
        foreach(_dir ${GGML_DLL_SEARCH_DIRS})
            if(EXISTS "${_dir}/${_dll}")
                configure_file("${_dir}/${_dll}" "${CMAKE_BINARY_DIR}/${_dll}" COPYONLY)
                message(STATUS "Staging DLL: ${_dll}")
                break()
            endif()
        endforeach()
    endforeach()
endif()

# Install targets
install(TARGETS gguf_loader text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3tts_shared qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/tts_transformer.h
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h
    src/audio_tokenizer_decoder.h
    src/mimi_encoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer --model models/qwen3-tts-0.6b-f16.gguf
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer --model models/qwen3-tts-0.6b-f16.gguf
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder --tokenizer models/qwen3-tts-tokenizer-f16-f16.gguf --codes reference/det_speech_codes.bin --reference reference/det_decoded_audio.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME batch_test
    COMMAND test_batch --model models/qwen3-tts-0.6b-f16.gguf --max-len 32
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME codebook_test
    COMMAND test_codebook --tokenizer models/qwen3-tts-tokenizer-f16-f16.gguf
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME mimi_encoder_test
    COMMAND test_mimi_encoder
        --tokenizer models/qwen3-tts-tokenizer-f16-f16.gguf
        --audio clone.wav
        --ref-codes reference/mimi_enc_py_codes.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

option(QWEN3_TTS_NPU_TESTS "Build experimental native Android/Hexagon validation tools" OFF)
if(QWEN3_TTS_NPU_TESTS)
    add_subdirectory(tools/npu)
endif()
