if(NOT DEFINED CLANG_FORMAT)
    message(FATAL_ERROR "CLANG_FORMAT is not set")
endif()

if(NOT DEFINED SOURCES_FILE)
    message(FATAL_ERROR "SOURCES_FILE is not set")
endif()

if(NOT DEFINED MODE)
    message(FATAL_ERROR "MODE is not set")
endif()

file(STRINGS "${SOURCES_FILE}" SOURCES)

if(NOT SOURCES)
    message(STATUS "No sources to format")
    return()
endif()

if(MODE STREQUAL "format")
    execute_process(
        COMMAND "${CLANG_FORMAT}" -i --style=file ${SOURCES}
        RESULT_VARIABLE result
    )
elseif(MODE STREQUAL "check")
    execute_process(
        COMMAND "${CLANG_FORMAT}" --dry-run --Werror --style=file ${SOURCES}
        RESULT_VARIABLE result
    )
else()
    message(FATAL_ERROR "Unknown clang-format mode: ${MODE}")
endif()

if(NOT result EQUAL 0)
    message(FATAL_ERROR "clang-format failed")
endif()
