cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

project(Vellum
    VERSION 3.0.0
    DESCRIPTION "Vellum Native Modern C++17 Backend & WebView2 Desktop Host"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT WIN32)
    message(FATAL_ERROR "Vellum currently targets Windows x64 (MSVC).")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Vellum requires a 64-bit (x64) build. Configure with `-A x64`.")
endif()

include(FetchContent)

# ------------------------------------------------------------------------------
# 1. cpp-httplib (header-only HTTP server & outbound client)
# ------------------------------------------------------------------------------
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZSTD_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_COMPILE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_INSTALL OFF CACHE BOOL "" FORCE)
set(HTTPLIB_TEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    httplib
    URL https://github.com/yhirose/cpp-httplib/archive/refs/tags/v0.67.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(httplib)

# ------------------------------------------------------------------------------
# 2. nlohmann/json (modern C++ JSON serialization)
# ------------------------------------------------------------------------------
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ------------------------------------------------------------------------------
# 3. Microsoft Edge WebView2 SDK headers + webview/webview (zero vcpkg/nuget)
# ------------------------------------------------------------------------------
set(MANGAFLOW_WEBVIEW2_VERSION "1.0.2903.40" CACHE STRING "Microsoft.Web.WebView2 SDK version")

FetchContent_Declare(
    microsoft_web_webview2
    URL "https://api.nuget.org/v3-flatcontainer/microsoft.web.webview2/${MANGAFLOW_WEBVIEW2_VERSION}/microsoft.web.webview2.${MANGAFLOW_WEBVIEW2_VERSION}.nupkg"
    DOWNLOAD_NAME "Microsoft.Web.WebView2.${MANGAFLOW_WEBVIEW2_VERSION}.zip"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(microsoft_web_webview2)
set(MSWebView2_ROOT "${microsoft_web_webview2_SOURCE_DIR}" CACHE PATH "Path to extracted Microsoft.Web.WebView2 package" FORCE)

set(WEBVIEW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
set(WEBVIEW_BUILD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(WEBVIEW_INSTALL_TARGETS OFF CACHE BOOL "" FORCE)
set(WEBVIEW_ENABLE_CHECKS OFF CACHE BOOL "" FORCE)
set(WEBVIEW_ENABLE_PACKAGING OFF CACHE BOOL "" FORCE)
set(WEBVIEW_USE_BUILTIN_MSWEBVIEW2 ON CACHE BOOL "" FORCE)

FetchContent_Declare(
    webview
    URL https://github.com/webview/webview/archive/refs/tags/0.12.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(webview)

# ------------------------------------------------------------------------------
# 4. Official Prebuilt ONNX Runtime (win-x64: CUDA [Universal default] / DML / CPU)
# ------------------------------------------------------------------------------
if(DEFINED MANGAFLOW_ORT_BACKEND AND NOT DEFINED VELLUM_ORT_BACKEND)
    set(VELLUM_ORT_BACKEND "${MANGAFLOW_ORT_BACKEND}" CACHE STRING "ONNX Runtime prebuilt backend: CUDA (universal default with safe CPU fallback), DML, CPU, or AUTO")
else()
    set(VELLUM_ORT_BACKEND "CUDA" CACHE STRING "ONNX Runtime prebuilt backend: CUDA (universal default with safe CPU fallback), DML, CPU, or AUTO")
endif()
set_property(CACHE VELLUM_ORT_BACKEND PROPERTY STRINGS CUDA DML DIRECTML CPU AUTO)
set(MANGAFLOW_ORT_URL "" CACHE STRING "Optional custom URL to an official onnxruntime-win-x64 zip/nupkg archive")

string(TOUPPER "${VELLUM_ORT_BACKEND}" _RESOLVED_ORT_BACKEND)
if(_RESOLVED_ORT_BACKEND STREQUAL "DML")
    set(_RESOLVED_ORT_BACKEND "DIRECTML")
endif()
if(_RESOLVED_ORT_BACKEND STREQUAL "AUTO")
    # Default Universal build uses Microsoft.ML.OnnxRuntime.Gpu.Windows with dynamic VRAM + safe CPU fallback
    set(_RESOLVED_ORT_BACKEND "CUDA")
    message(STATUS "[Vellum] AUTO backend -> selecting Microsoft.ML.OnnxRuntime.Gpu (CUDA + safe CPU fallback)")
endif()

if(MANGAFLOW_ORT_URL)
    set(_ORT_DOWNLOAD_URL "${MANGAFLOW_ORT_URL}")
    set(_ORT_DOWNLOAD_NAME "onnxruntime-win-x64-custom.zip")
elseif(_RESOLVED_ORT_BACKEND STREQUAL "DIRECTML")
    set(_ORT_VERSION "1.24.4")
    set(_ORT_DOWNLOAD_URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/${_ORT_VERSION}/microsoft.ml.onnxruntime.directml.${_ORT_VERSION}.nupkg")
    set(_ORT_DOWNLOAD_NAME "onnxruntime-win-x64-directml-${_ORT_VERSION}.zip")
elseif(_RESOLVED_ORT_BACKEND STREQUAL "CUDA")
    set(_ORT_VERSION "1.24.4")
    set(_ORT_DOWNLOAD_URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.gpu.windows/${_ORT_VERSION}/microsoft.ml.onnxruntime.gpu.windows.${_ORT_VERSION}.nupkg")
    set(_ORT_DOWNLOAD_NAME "onnxruntime-win-x64-gpu-${_ORT_VERSION}.zip")
elseif(_RESOLVED_ORT_BACKEND STREQUAL "CPU")
    set(_ORT_VERSION "1.24.4")
    set(_ORT_DOWNLOAD_URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${_ORT_VERSION}/microsoft.ml.onnxruntime.${_ORT_VERSION}.nupkg")
    set(_ORT_DOWNLOAD_NAME "onnxruntime-win-x64-${_ORT_VERSION}.zip")
else()
    message(FATAL_ERROR "Invalid VELLUM_ORT_BACKEND='${VELLUM_ORT_BACKEND}'. Choose DML, DIRECTML, CUDA, CPU, or AUTO.")
endif()

FetchContent_Declare(
    onnxruntime_prebuilt
    URL "${_ORT_DOWNLOAD_URL}"
    DOWNLOAD_NAME "${_ORT_DOWNLOAD_NAME}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime_prebuilt)

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        "${onnxruntime_prebuilt_SOURCE_DIR}/build/native/include"
        "${onnxruntime_prebuilt_SOURCE_DIR}/buildTransitive/native/include"
        "${onnxruntime_prebuilt_SOURCE_DIR}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(ONNXRUNTIME_IMPLIB
    NAMES onnxruntime
    PATHS
        "${onnxruntime_prebuilt_SOURCE_DIR}/runtimes/win-x64/native"
        "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_IMPLIB}" DIRECTORY)
set(ONNXRUNTIME_DLL "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
if(NOT EXISTS "${ONNXRUNTIME_DLL}")
    message(FATAL_ERROR "onnxruntime.dll not found in ${ONNXRUNTIME_LIB_DIR}")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_IMPLIB}"
    IMPORTED_LOCATION "${ONNXRUNTIME_DLL}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

set(MANGAFLOW_HAS_DML 0)
if(EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/dml_provider_factory.h")
    set(MANGAFLOW_HAS_DML 1)
endif()

# ------------------------------------------------------------------------------
# 5. miniz (amalgamated ZIP / zlib single-pair) & stb (stb_image + stb_image_write)
# ------------------------------------------------------------------------------
FetchContent_Declare(
    miniz_prebuilt
    URL https://github.com/richgel999/miniz/releases/download/3.1.2/miniz-3.1.2.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(miniz_prebuilt)

add_library(miniz_static STATIC
    "${miniz_prebuilt_SOURCE_DIR}/miniz.c"
)
target_include_directories(miniz_static PUBLIC "${miniz_prebuilt_SOURCE_DIR}")
if(MSVC)
    target_compile_definitions(miniz_static PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(miniz_static PRIVATE
        /W0
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
    )
    set_target_properties(miniz_static PROPERTIES
        STATIC_LIBRARY_OPTIONS "$<$<CONFIG:Release>:/LTCG>"
    )
endif()

FetchContent_Declare(
    stb_headers
    URL https://github.com/nothings/stb/archive/refs/heads/master.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(stb_headers)

add_library(stb::headers INTERFACE IMPORTED)
set_target_properties(stb::headers PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${stb_headers_SOURCE_DIR}"
)

message(STATUS "[Vellum] ONNX Runtime backend : ${VELLUM_ORT_BACKEND} (resolved: ${_RESOLVED_ORT_BACKEND})")
message(STATUS "[Vellum] ONNX Runtime include : ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "[Vellum] ONNX Runtime lib dir : ${ONNXRUNTIME_LIB_DIR}")
message(STATUS "[Vellum] DirectML header      : ${MANGAFLOW_HAS_DML}")
message(STATUS "[Vellum] miniz source dir     : ${miniz_prebuilt_SOURCE_DIR}")
message(STATUS "[Vellum] stb include dir      : ${stb_headers_SOURCE_DIR}")

# ------------------------------------------------------------------------------
# 6. Native Windows GUI Executable (WIN32 subsystem = zero black console window)
# ------------------------------------------------------------------------------
set(MANGAFLOW_RC_FILE "")
set(MANGAFLOW_ICON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../packaging/icon.ico")
if(EXISTS "${MANGAFLOW_ICON_PATH}")
    file(TO_NATIVE_PATH "${MANGAFLOW_ICON_PATH}" _ICON_NATIVE)
    string(REPLACE "\\" "\\\\" MANGAFLOW_ICON_RC_PATH "${_ICON_NATIVE}")
    set(MANGAFLOW_RC_FILE "${CMAKE_CURRENT_BINARY_DIR}/app.rc")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/src/app.rc.in"
        "${MANGAFLOW_RC_FILE}"
        @ONLY
    )
endif()

add_executable(MangaFlowAI WIN32
    src/main.cpp
    src/app_context.hpp
    src/app_context.cpp
    src/routes/routes_system.hpp
    src/routes/routes_system.cpp
    src/routes/routes_pipeline.hpp
    src/routes/routes_pipeline.cpp
    src/routes/routes_projects.hpp
    src/routes/routes_projects.cpp
    src/routes/routes_export.hpp
    src/routes/routes_export.cpp
    src/settings_manager.cpp
    src/font_manager.cpp
    src/exporter.cpp
    src/psd_writer.hpp
    src/psd_writer.cpp
    src/detector/detector_types.hpp
    src/detector/detector_engine.hpp
    src/detector/detector_engine.cpp
    src/detector/cv_refinement.hpp
    src/detector/cv_refinement.cpp
    src/detector/cv_typography.hpp
    src/detector/cv_typography.cpp
    src/detector/cv_geometry.hpp
    src/detector/cv_geometry.cpp
    src/detector/geometry_postproc.hpp
    src/detector/geometry_postproc.cpp
    src/detector/math_det_ops.hpp
    src/detector/math_det_ops.cpp
    src/detector/yolo_seg_backend.hpp
    src/detector/yolo_seg_backend.cpp
    src/detector/adaptive_tiler.hpp
    src/detector/adaptive_tiler.cpp
    src/detector/export_artifacts.hpp
    src/detector/export_artifacts.cpp
    src/pipeline/ort_provider.cpp
    src/pipeline/model_manager.cpp
    src/pipeline/detector.cpp
    src/pipeline/math_ocr_ops.hpp
    src/pipeline/math_ocr_ops.cpp
    src/pipeline/ocr.cpp
    src/pipeline/math_inpaint_ops.hpp
    src/pipeline/math_inpaint_ops.cpp
    src/pipeline/inpainter.cpp
    src/pipeline/math_nlp_qc.hpp
    src/pipeline/math_nlp_qc.cpp
    src/pipeline/translator.cpp
    ${MANGAFLOW_RC_FILE}
)

set_target_properties(MangaFlowAI PROPERTIES
    OUTPUT_NAME "Vellum"
)

target_compile_features(MangaFlowAI PRIVATE cxx_std_17)

target_compile_definitions(MangaFlowAI PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0A00
    UNICODE
    _UNICODE
    MANGAFLOW_HAS_DML=${MANGAFLOW_HAS_DML}
)

if(MSVC)
    target_compile_options(MangaFlowAI PRIVATE
        /W4
        /utf-8
        /permissive-
        /EHsc
        /MP
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
    )
    target_link_options(MangaFlowAI PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
endif()

target_link_libraries(MangaFlowAI PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    webview::core_static
    onnxruntime::onnxruntime
    miniz_static
    stb::headers
    ws2_32
    dwmapi
    dxgi
    winhttp
    crypt32
    ole32
    shell32
    uuid
)

# Copy ONNX Runtime DLLs next to Vellum.exe after every build
file(GLOB ONNXRUNTIME_RUNTIME_DLLS "${ONNXRUNTIME_LIB_DIR}/*.dll")
foreach(_dll IN LISTS ONNXRUNTIME_RUNTIME_DLLS)
    add_custom_command(TARGET MangaFlowAI POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_dll}"
            "$<TARGET_FILE_DIR:MangaFlowAI>"
        COMMENT "Copying ONNX Runtime DLL: ${_dll}"
    )
endforeach()

# Copy self-contained NVIDIA CUDA 12 + cuDNN 9 redistributable DLLs (if present in build_deps/cuda_runtime/bin)
set(VELLUM_CUDA_RUNTIME_BIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build_deps/cuda_runtime/bin")
if(_RESOLVED_ORT_BACKEND STREQUAL "CUDA" AND EXISTS "${VELLUM_CUDA_RUNTIME_BIN_DIR}")
    file(GLOB VELLUM_CUDA_RUNTIME_DLLS "${VELLUM_CUDA_RUNTIME_BIN_DIR}/*.dll")
    foreach(_cuda_dll IN LISTS VELLUM_CUDA_RUNTIME_DLLS)
        add_custom_command(TARGET MangaFlowAI POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_cuda_dll}"
                "$<TARGET_FILE_DIR:MangaFlowAI>"
            COMMENT "Copying CUDA Runtime DLL: ${_cuda_dll}"
        )
    endforeach()
endif()

# ------------------------------------------------------------------------------
# 7. Root One-Click Native Launcher (`Manga/Vellum.exe`)
# ------------------------------------------------------------------------------
add_executable(MangaFlowCppLauncher WIN32
    src/launcher.cpp
    ${MANGAFLOW_RC_FILE}
)

set_target_properties(MangaFlowCppLauncher PROPERTIES
    OUTPUT_NAME "Vellum-Launcher"
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
)

target_compile_features(MangaFlowCppLauncher PRIVATE cxx_std_17)

target_compile_definitions(MangaFlowCppLauncher PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0A00
    UNICODE
    _UNICODE
)

if(MSVC)
    target_compile_options(MangaFlowCppLauncher PRIVATE
        /W4
        /utf-8
        /permissive-
        /EHsc
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
    )
    target_link_options(MangaFlowCppLauncher PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
endif()

# ------------------------------------------------------------------------------
# 8. Standalone Detector Engine v2 Benchmark CLI (`vellum_det_bench.exe`)
# ------------------------------------------------------------------------------
add_executable(vellum_det_bench
    src/detector/vellum_det_bench.cpp
    src/detector/detector_types.hpp
    src/detector/detector_engine.hpp
    src/detector/detector_engine.cpp
    src/detector/cv_refinement.hpp
    src/detector/cv_refinement.cpp
    src/detector/cv_typography.hpp
    src/detector/cv_typography.cpp
    src/detector/cv_geometry.hpp
    src/detector/cv_geometry.cpp
    src/detector/geometry_postproc.hpp
    src/detector/geometry_postproc.cpp
    src/detector/math_det_ops.hpp
    src/detector/math_det_ops.cpp
    src/detector/yolo_seg_backend.hpp
    src/detector/yolo_seg_backend.cpp
    src/detector/adaptive_tiler.hpp
    src/detector/adaptive_tiler.cpp
    src/detector/export_artifacts.hpp
    src/detector/export_artifacts.cpp
    src/pipeline/detector.cpp
    src/pipeline/math_ocr_ops.hpp
    src/pipeline/math_ocr_ops.cpp
    src/pipeline/ort_provider.cpp
    src/pipeline/model_manager.cpp
)

target_compile_features(vellum_det_bench PRIVATE cxx_std_17)

target_compile_definitions(vellum_det_bench PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0A00
    MANGAFLOW_HAS_DML=${MANGAFLOW_HAS_DML}
)

if(MSVC)
    target_compile_options(vellum_det_bench PRIVATE
        /W4
        /utf-8
        /permissive-
        /EHsc
        /MP
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
    )
    target_link_options(vellum_det_bench PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
endif()

target_link_libraries(vellum_det_bench PRIVATE
    nlohmann_json::nlohmann_json
    onnxruntime::onnxruntime
    stb::headers
    psapi
    dxgi
    winhttp
    ws2_32
)

# ------------------------------------------------------------------------------
# 9. Standalone Layered PSD Writer Unit Test (`test_psd_writer.exe`)
# ------------------------------------------------------------------------------
add_executable(test_psd_writer
    tests/test_psd_writer.cpp
    src/psd_writer.hpp
    src/psd_writer.cpp
)

target_compile_features(test_psd_writer PRIVATE cxx_std_17)

if(MSVC)
    target_compile_options(test_psd_writer PRIVATE
        /W4
        /utf-8
        /permissive-
        /EHsc
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/GL>
        $<$<CONFIG:Release>:/Gy>
        $<$<CONFIG:Release>:/Gw>
    )
    target_link_options(test_psd_writer PRIVATE
        $<$<CONFIG:Release>:/LTCG>
        $<$<CONFIG:Release>:/INCREMENTAL:NO>
        $<$<CONFIG:Release>:/OPT:REF>
        $<$<CONFIG:Release>:/OPT:ICF>
    )
endif()

target_link_libraries(test_psd_writer PRIVATE
    stb::headers
)

# ------------------------------------------------------------------------------
# 10. v2.5 Studio Core Execution-Based Audit & Regression Suite (`test_v25_audit.exe`)
# ------------------------------------------------------------------------------
add_executable(test_v25_audit
    tests/test_v25_audit.cpp
    src/settings_manager.cpp
    src/exporter.cpp
    src/detector/cv_refinement.cpp
    src/detector/cv_typography.cpp
    src/detector/cv_geometry.cpp
    src/detector/geometry_postproc.cpp
    src/detector/math_det_ops.cpp
    src/pipeline/ort_provider.cpp
    src/pipeline/model_manager.cpp
    src/pipeline/detector.cpp
    src/pipeline/math_ocr_ops.hpp
    src/pipeline/math_ocr_ops.cpp
    src/pipeline/ocr.cpp
    src/pipeline/math_inpaint_ops.hpp
    src/pipeline/math_inpaint_ops.cpp
    src/pipeline/inpainter.cpp
    src/pipeline/math_nlp_qc.hpp
    src/pipeline/math_nlp_qc.cpp
    src/pipeline/translator.cpp
)

target_compile_features(test_v25_audit PRIVATE cxx_std_17)

target_compile_definitions(test_v25_audit PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _WIN32_WINNT=0x0A00
    MANGAFLOW_HAS_DML=${MANGAFLOW_HAS_DML}
)

if(MSVC)
    target_compile_options(test_v25_audit PRIVATE
        /W4
        /utf-8
        /permissive-
        /EHsc
        /MP
        $<$<CONFIG:Release>:/O2>
    )
endif()

target_link_libraries(test_v25_audit PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    onnxruntime::onnxruntime
    miniz_static
    stb::headers
    ws2_32
    dxgi
    winhttp
    crypt32
)
