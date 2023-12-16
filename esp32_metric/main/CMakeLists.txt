SET(METRIC_SOURCE
    model_metric.cpp
    test_data.c )


idf_component_register(SRCS ${METRIC_SOURCE}
                    INCLUDE_DIRS "")
