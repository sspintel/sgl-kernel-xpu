set(FMHA_XE35_HEAD_DIMS 64 128)

set(FMHA_XE35_DECODE_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/flash_attentionXe35_decode_kernel.cpp.in")

foreach(HEAD_DIM ${FMHA_XE35_HEAD_DIMS})
    set(GENERATED_DECODE_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/flash_attentionXe35_decode_${HEAD_DIM}.cpp")
    configure_file(${FMHA_XE35_DECODE_TEMPLATE} ${GENERATED_DECODE_FILE} @ONLY)
    list(APPEND device_cpp_xe35 ${GENERATED_DECODE_FILE})
endforeach()
