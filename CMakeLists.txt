# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("synth")

file(GLOB stk_include
     "synth/stk/*.h"
     "synth/stk/*.cpp"
)

add_executable (synth 
	synth/main.cpp
    ${stk_include}
)

target_compile_definitions(synth PRIVATE __OS_WINDOWS__ __LITTLE_ENDIAN__)
