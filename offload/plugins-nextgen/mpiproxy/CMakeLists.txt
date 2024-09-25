# Looking for MPI...
find_package(MPI QUIET)

if(NOT(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  message(STATUS "Not building MPI offloading plugin: only support MPI in Linux x86_64 or ppc64le hosts.")
  return()
elseif(NOT MPI_CXX_FOUND)
  message(STATUS "Not building MPI offloading plugin: MPI not found in system.")
  return()
endif()

message(STATUS "Building MPI Proxy offloading plugin.")

# Event System
add_subdirectory(event_system)

# Host Side Plugin
add_subdirectory(host-plugin)

# Configure testing for the MPI plugin.
list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.mpiproxy")
# Report to the parent scope that we are building a plugin for MPI.
set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)

# Define the target specific triples and ELF machine values.
set(LIBOMPTARGET_SYSTEM_TARGETS
    "${LIBOMPTARGET_SYSTEM_TARGETS} x86_64-pc-linux-gnu-mpi nvptx64-nvidia-cuda-mpi" PARENT_SCOPE) 


# Remote Plugin Manager
message(STATUS "Building the remote-proxy-device")

set(LIBOMPTARGET_ALL_REMOTE_PLUGIN_TARGETS cuda host)
set(LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD "all" CACHE STRING
    "Semicolon-separated list of plugins to use: cuda, amdgpu, host or \"all\".")

if(LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD STREQUAL "all")
  set(LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD ${LIBOMPTARGET_ALL_REMOTE_PLUGIN_TARGETS})
endif()

set(REMOTE_MPI_ENUM_PLUGIN_TARGETS "")
foreach(plugin IN LISTS LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD)
  if(NOT(plugin MATCHES "mpiproxy"))
    set(REMOTE_MPI_ENUM_PLUGIN_TARGETS
        "${REMOTE_MPI_ENUM_PLUGIN_TARGETS}PLUGIN_TARGET(${plugin})\n")
  endif()
endforeach()
string(STRIP ${REMOTE_MPI_ENUM_PLUGIN_TARGETS} REMOTE_MPI_ENUM_PLUGIN_TARGETS)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/RemoteTargets.def.in
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}/Shared/RemoteTargets.def
)

llvm_add_tool(OPENMP remote-proxy-device
  src/ProxyDevice.cpp
  src/RemotePluginManager.cpp
  ${LIBOMPTARGET_SRC_DIR}/OpenMP/OMPT/Callback.cpp
)

llvm_update_compile_flags(remote-proxy-device)

target_link_libraries(remote-proxy-device PRIVATE
  EventSystem
  LLVMSupport
  omp
)

add_dependencies(remote-proxy-device omp)

target_include_directories(remote-proxy-device PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

foreach(plugin IN LISTS LIBOMPTARGET_REMOTE_PLUGINS_TO_BUILD)
  if(NOT(plugin MATCHES "mpiproxy"))
    target_link_libraries(remote-proxy-device PRIVATE omptarget.rtl.${plugin})
    add_dependencies(remote-proxy-device omptarget.rtl.${plugin})
  endif()
endforeach()

# Set C++20 as the target standard for this plugin.
set_target_properties(remote-proxy-device
                      PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED ON)

target_compile_definitions(remote-proxy-device PRIVATE
                           TARGET_NAME=remote-proxy-device
                           DEBUG_PREFIX="REMOTE PLUGIN MANAGER")
