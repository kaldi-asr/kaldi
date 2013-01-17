#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey);
#                 Arnab Ghoshal, Karel Vesely

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Parse command-line options.
# To be sourced by another script (as in ". parse_options.sh").
# Option format is: --option-name arg
# and shell variable "option_name" gets set to value "arg."
# The exception is --help, which takes no arguments, but prints the 
# $help_message variable (if defined).


###
### This function will import options from a config file.
### It checks that the option is defined in the top level script.
###
function import_config {
  # Import the config
  [ -z "$config" ] && echo "import_config: \$config was not set" && exit 1
  # Check that the file exists
  [ ! -f $config ] && echo "Cannot read config $config" && exit 1
  # Import the config options
  while read line; do
    #Remove white chars so we can detect empty lines or simple comments
    line_no_wchar=$(echo $line | sed 's|\s||g') 
    [ "${line_no_wchar}" == "" ] && continue      #ignore empty lines
    [ "${line_no_wchar:0:1}" == "#" ] && continue #lines starts by #, ignore comments
    #Get the name of the option
    name=$(echo $line | sed -e 's|^\s*\([0-9a-zA-Z_\-]*\)=.*$|\1|')
    [ -z "$name" ] && echo "$0: cannot locate option name in config line '$line' at $config" && exit 1
    eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $name at $config" && exit 1;
    #run the original line as it is contains an expected option
    eval "$line"
  done < $config
} 



###
### The --config file options have lower priority to command line 
### options, so we need to import them first...
###

# Now import all the configs specified by command-line, in left-to-right order
for ((n=1; n<$#; n++)); do
  if [ "${!n}" == "--config" ]; then
    n_plus1=$((n+1))
    config=${!n_plus1}
    import_config
  fi
done


###
### No we process the command line options
###
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    # If the enclosing script is called with --help option, print the help 
    # message and exit.  Scripts should put help messages in $help_message
  --help) if [ -z "$help_message" ]; then echo "No help found.";
	  else printf "$help_message\n"; fi; 
	  exit 0 ;; 
    # If the first command-line argument begins with "--" (e.g. --foo-bar), 
    # then work out the variable name as $name, which will equal "foo_bar".
  --*) name=`echo "$1" | sed s/^--// | sed s/-/_/g`; 
    # Next we test whether the variable in question is undefned-- if so it's 
    # an invalid option and we die.  Note: $0 evaluates to the name of the 
    # enclosing script.
    # The test [ -z ${foo_bar+xxx} ] will return true if the variable foo_bar
    # is undefined.  We then have to wrap this test inside "eval" because 
    # foo_bar is itself inside a variable ($name).
      eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $1" && exit 1;
      
      oldval="`eval echo \\$$name`";
    # Work out whether we seem to be expecting a Boolean argument.
      if [ "$oldval" == "true" ] || [ "$oldval" == "false" ]; then 
	was_bool=true;
      else 
	was_bool=false;
      fi

    # Set the variable to the right value-- the escaped quotes make it work if
    # the option had spaces, like --cmd "queue.pl -sync y"
      eval $name=\"$2\"; 
        
    # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": --$name $2"
        exit 1;
      fi
      shift 2;
      ;;
  *) break;
  esac
done


# Check for an empty argument to the --cmd option, which can easily occur as a 
# result of scripting errors.
[ ! -z "${cmd+xxx}" ] && [ -z "$cmd" ] && echo "$0: empty argument to --cmd option" && exit 1;


true; # so this script returns code zero.

