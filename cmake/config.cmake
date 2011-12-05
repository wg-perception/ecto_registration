set(registration_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(registration_LIBS ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libregistration.so)
configure_file(cmake/registrationConfig.cmake.in
  ${CMAKE_BINARY_DIR}/registrationConfig.cmake
  @ONLY
)
configure_file(cmake/registrationConfig-version.cmake.in
  ${CMAKE_BINARY_DIR}/registrationConfig-version.cmake
  @ONLY
)

