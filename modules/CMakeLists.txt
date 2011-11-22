include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# currently just a dummy, we'll want to add sources here for estimation
ectomodule(ecto_registration
  module.cpp
  )

link_ecto(ecto_registration
  ${OpenCV_LIBS}
  )

install_ecto_module(ecto_registration)
