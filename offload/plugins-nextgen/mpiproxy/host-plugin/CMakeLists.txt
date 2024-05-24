# Create the library and add the default arguments.
add_target_library(omptarget.rtl.mpiproxy MPI)

target_sources(omptarget.rtl.mpiproxy PRIVATE
  src/rtl.cpp
)

target_link_libraries(omptarget.rtl.mpiproxy PRIVATE
  EventSystem
)

# Add include directories
target_include_directories(omptarget.rtl.mpiproxy PRIVATE
                           ${LIBOMPTARGET_INCLUDE_DIR})

# Set C++20 as the target standard for this plugin.
set_target_properties(omptarget.rtl.mpiproxy
                      PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED ON)
