#!/bin/bash
# Copyright Daniel Povey, 2012.  Apache 2.0.

# Parse command-line options-- to be sourced by another script (as in ". parse_options.sh")
# option format is:
# --option-name arg
# and shell variable "option_name" gets set to value "arg."


# The following assignment allows the --config variable to be specified
# in all cases.
[ -z "$config" ] && config=

while true; do
 case "$1" in
        # If the first command-line argument begins with "--" (e.g. --foo-bar), then work out  
        # the variable name as $name, which will equal "foo_bar".
  --*) name=`echo "$1" | sed s/^--// | sed s/-/_/g`; 
        # Next we test whether the variable in question is undefned-- if so it's an
        # invalid option and we die.  Note: $0 evaluates to the name of the enclosing
        # script.
        # The test [ -z ${foo_bar+xxx} ] will return true if the variable foo_bar
        # is undefined.  We then have to wrap this test inside "eval" because foo_bar is
        # itself inside a variable ($name).
        eval '[ -z "${'$name'+xxx}" ] && echo "$0: invalid option $1" && exit 1;'
        # Set the variable to the right value-- the escaped quotes make it work if
        # the option had spaces, like --cmd "queue.pl -sync y"
        eval $name=\"$2\"; shift 2;;
  *) break;
 esac
done

[ ! -z "$config" ] && . $config # Override any of the options, if --config was specified.

# Check for an empty argument to the --cmd option, which can easily occur as a result
# of scripting errors.
[ ! -z "${cmd+xxx}" ] && [ -z "$cmd" ] && echo "$0: empty argument to --cmd option" && exit 1;

true; # so this script returns code zero.

