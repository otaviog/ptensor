add_executable(face_detector main.cpp)
ptensor_target_options(face_detector "recog")
target_link_libraries(face_detector PRIVATE ptensor_recog ptensor_op ptensor_media ptensor_io ptensor_viz CLI11::CLI11)
