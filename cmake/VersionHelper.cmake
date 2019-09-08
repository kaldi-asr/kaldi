function(get_version)

execute_process(COMMAND git --format='%H' -n1 OUTPUT_VARIABLE version_commit)
execute_process(COMMAND git rev-list ${version_commit}..HEAD OUTPUT_VARIABLE patch_number)

set(version "??")

endfunction()
