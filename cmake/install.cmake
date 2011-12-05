set(prefix registration-${registration_VERSION})
install(DIRECTORY ${registration_SOURCE_DIR}/include/
  DESTINATION include/${prefix}
  COMPONENT main
  )

#install the unix_install
install(DIRECTORY ${registration_BINARY_DIR}/share/
  DESTINATION share/${prefix}
  COMPONENT main
  )

