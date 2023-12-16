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

    SET(COMPONENT_SRCS
    model_v2.c
    test_data2.c
    )

    SET(INCLUDE_DIR
    include_v2
    )

endif()

idf_component_register(SRCS ${COMPONENT_SRCS}
                        INCLUDE_DIRS ${INCLUDE_DIR})