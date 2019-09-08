cmake_minimum_required(VERSION 2.8)

project(BASICTHRIFT)
aux_source_directory(./thrift_interface DIR_SRC)
add_executable(master ${DIR_SRC})
target_link_libraries(master -lthrift)
