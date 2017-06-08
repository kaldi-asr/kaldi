#! /bin/bash

function usage()
{
    cmnd=$(basename $0);
    cat<<EOF

$cmnd - removes sentence start/end symbols

USAGE:
       $cmnd [options]

OPTIONS:
       -h        Show this message

EOF
}

# Parse options
while getopts h OPT; do
    case "$OPT" in
        h)
            usage >&2;
            exit 0;
            ;;
    esac
done

sed 's/<s>//g' | sed 's/<\/s>//g' | sed 's/^ *//' | sed 's/ *$//' | sed '/^$/d'

