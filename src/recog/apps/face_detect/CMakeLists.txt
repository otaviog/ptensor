add_executable(face_detector main)
ptensor_target_options(face_detector "recog")
target_link_libraries(face_detector PRIVATE ptensor_recog ptensor_media ptensor_io CLI11::CLI11)
