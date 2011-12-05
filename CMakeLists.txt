cmake_minimum_required(VERSION 2.8)
project(ecto_registration)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
  "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
  FORCE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#add_definitions("-Wall -Werror")

find_package(Boost REQUIRED)
find_package(Eigen REQUIRED)
find_package(g2o 0.1.0 REQUIRED)
find_package(OpenCV 2.3.1 EXACT REQUIRED)
find_package(ecto 0.1.0 REQUIRED)
find_package(image_pipeline 0.1.0 REQUIRED)

# where to find the python cells
set_ecto_install_package_name(ecto_registration)

configure_file(python/__init__.py.in
  ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/__init__.py
  @ONLY)

ecto_python_env_gen(${CMAKE_BINARY_DIR}/lib ${CMAKE_CURRENT_SOURCE_DIR}/python)

include_directories(
  include
  ${EIGEN_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${image_pipeline_INCLUDE_DIRS}
  ${g2o_INCLUDE_DIRS}
  )

add_subdirectory(src)

option(BUILD_ECTO_CELLS "Build ecto cells" ON)
if(BUILD_ECTO_CELLS)
  add_subdirectory(cells)
endif()

option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

#these setup the lib to be used by others
include(cmake/install.cmake)
include(cmake/config.cmake)
