set(FMHA_XE35_HEAD_DIMS 64 128)

set(FMHA_XE35_PREFILL_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/flash_attentionXe35_prefill_kernel.cpp.in")

foreach(HEAD_DIM ${FMHA_XE35_HEAD_DIMS})
    set(GENERATED_PREFILL_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/flash_attentionXe35_prefill_${HEAD_DIM}.cpp")
    configure_file(${FMHA_XE35_PREFILL_TEMPLATE} ${GENERATED_PREFILL_FILE} @ONLY)
    list(APPEND device_cpp_xe35 ${GENERATED_PREFILL_FILE})
endforeach()
