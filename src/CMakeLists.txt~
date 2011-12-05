include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(registration SHARED
  pe3d.cpp
  stereo_polish.cpp
  )

target_link_libraries(registration
  ${OpenCV_LIBS}
  ${image_pipeline_LIBS}
  ${g2o_LIBS}
  )

  set_target_properties(registration PROPERTIES
    VERSION ${registration_VERSION}
    SOVERSION ${registration_SOVERSION}
    )

#install the main shared lib
install(TARGETS registration
  DESTINATION lib
  COMPONENT main
  )

