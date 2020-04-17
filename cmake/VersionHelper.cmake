function(get_version)
    file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/.version version)
    string(STRIP ${version} version)
    execute_process(COMMAND git log -n1 --format=%H src/.version
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE version_commit
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND git rev-list --count "${version_commit}..HEAD"
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    OUTPUT_VARIABLE patch_number)
    string(STRIP ${patch_number} patch_number)

    set(KALDI_VERSION ${version} PARENT_SCOPE)
    set(KALDI_PATCH_NUMBER ${patch_number} PARENT_SCOPE)
endfunction()
