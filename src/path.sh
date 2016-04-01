# We are using BASH_SOURCE[0], because its set correctly even when the file
# is sourced.
this_script_path="$(readlink -f "${BASH_SOURCE[0]}")"
my_kaldi_src="$(dirname "$this_script_path")"
add_to_path="$(shopt -s failglob
               echo "$my_kaldi_src/"*bin/ |
                 (read -ra a; IFS=:; echo "${a[*]}"))" \
  && PATH="$add_to_path:$PATH"
unset -v this_script_path my_kaldi_src add_to_path
