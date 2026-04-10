cmake_minimum_required(VERSION 3.25)
project(braille-scope VERSION 1 LANGUAGES CXX)

find_package(Boost REQUIRED CONFIG)
find_package(PkgConfig)
pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3f)
pkg_check_modules(JACK REQUIRED IMPORTED_TARGET jack)
pkg_check_modules(SndFile REQUIRED IMPORTED_TARGET sndfile)

add_executable(braille-scope braille-scope.cpp)
target_compile_features(braille-scope PRIVATE cxx_std_23)
set_property(TARGET braille-scope PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(braille-scope
  PRIVATE Boost::boost PkgConfig::FFTW PkgConfig::JACK PkgConfig::SndFile
)
install(TARGETS braille-scope RUNTIME DESTINATION bin)
