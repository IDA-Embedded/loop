if(CONFIG_MODEL_VERSION_1_0)

    SET(COMPONENT_SRCS
        model_v1.c
        test_data.c
        )

    SET(INCLUDE_DIR
        include_v1
        )

endif()


if(CONFIG_MODEL_VERSION_2_0)


    if(CONFIG_USE_QUANTIZED_MODEL)

        SET(COMPONENT_SRCS
        model_quantized_v2.c
        test_quantized.c
        )

        SET(INCLUDE_DIR
        include_v2/model_v2_quan
        )


    else()

        SET(COMPONENT_SRCS
        model_v2.c
        test_data2.c
        )

        SET(INCLUDE_DIR
        include_v2/model_v2
        )

    endif()


endif()

idf_component_register(SRCS ${COMPONENT_SRCS}
                        INCLUDE_DIRS ${INCLUDE_DIR})