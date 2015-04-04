#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Vijayaditya Peddinti).  Apache 2.0.

# This script operates on alignment directories, such as exp/tri4a_ali

# Begin configuration section. 
extra_files= # specify addtional files in 'src-data-dir' to merge, ex. "file1 file2 ..."
ref_data_dir= # data directory to be used as reference for rearranging alignments
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 2 ]; then
  echo "Usage: combine_ali_dirs.sh [--extra-files 'file1 file2'] <dest-ali-dir> <src-ali-dir1> <src-ali-dir2> ..."
  echo "Note, files that don't appear in first source dir will not be added even if they appear in later ones."
  echo "Other than alignments, only files from the first src dir are copied."
  exit 1
fi

dest=$1;
shift;

first_src=$1;

rm -r $dest 2>/dev/null
mkdir -p $dest;

export LC_ALL=C

for dir in $*; do
  if [ ! -f $dir/ali.1.gz ]; then
    echo "$0: check if alignments (ali.*.gz) are present in $dir."
    exit 1;
  fi
done

for dir in $*; do
  for f in tree; do
    diff $first_src/$f $dir/$f 1>/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "$0: Cannot combine alignment directories with different $f files."
    fi
  done
done

for f in final.mdl tree cmvn_opts splice_opts num_jobs $extra_files; do
  if [ ! -f $first_src/$f ]; then
    echo "combine_ali_dir.sh: no such file $first_src/$f"
    exit 1;
  fi
  cp $first_src/$f $dest/
done

job_id=0
for dir in $*; do
  cur_num_jobs=$(cat $dir/num_jobs)
  for i in `seq 1 $cur_num_jobs`; do
    job_id=$((job_id + 1))
    mv $dir/ali.$i.gz $dest/ali.$job_id.gz
  done
done
echo $job_id > $dest/num_jobs

num_jobs=$job_id
if [ ! -z "$ref_data_dir" ]; then
  # redistribute the alignments into gz files so that each ali.*.gz file has 
  # same utterances as corresponding feats.scp in ref_data_dir
  temp_dir=$dest/temp1
  echo "Resplitting the ali.*.gz files to correspond to $ref_data_dir/split*/*/feats.scp files"
  mkdir -p $temp_dir
  cat <<EOF >$temp_dir/create_ali_utt_index.sh
   JOB=\$1
   ali_file=$dest/ali.\$JOB.gz
   gunzip -c \$ali_file | copy-int-vector ark:- ark,t:- | \
    awk -v p=\$ali_file '{printf "%s %s %s\n", \$1, p, NR}' > $temp_dir/ali_utt_index.\$JOB
EOF
  chmod +x $temp_dir/create_ali_utt_index.sh
  $decode_cmd -v PATH JOB=1:$num_jobs $temp_dir/ali_copy_int.JOB.log $temp_dir/create_ali_utt_index.sh JOB

  cat <<EOF >$temp_dir/create_new_ali.py

import sys, subprocess, glob, os

def ali_utt_index_filename(pattern):
  ali_utt_files = glob.glob('exp/tri5a_rvb_ali/temp1/ali_utt_index.*')
  if not len(ali_utt_files) > 0:
    raise Exception("ali_utt_index files are missing")
  ali_utt_files = " ".join(ali_utt_files)
  proc = subprocess.Popen("grep {0} -n  -e {1}".format(ali_utt_files, pattern).split(), stdout = subprocess.PIPE, stderr = subprocess.PIPE )
  out, err = proc.communicate()
  print out, err
  parts = out.split(":")
  return (parts[0], int(parts[1])) # returns the filename and line number

if __name__ == "__main__":
  utt2spk_filename = sys.argv[1]  
  command_filename = sys.argv[2]
  ali_gz_filename = sys.argv[3]

  utt_list=map(lambda x: x.split()[0], open(utt2spk_filename).readlines())
  file_name, line_number = ali_utt_index_filename(utt_list[0])
  parts = open(file_name).readlines()[line_number - 1].split()
  ali_file_name = parts[1]
  ali_line_number = int(parts[2])
  print ali_file_name, ali_line_number
  ali_utt_index = {}
  junk = map(lambda x: ali_utt_index.update({x.split()[0]: x.split()[1:]}), open(file_name).readlines())
  # read the next file too incase we cross into the other ali.*.gz file
  parts = file_name.split(".")
  parts[-1] = str(int(parts[-1]) + 1)
  next_file_name = ".".join(parts)
  commands = []
  try:
    junk = map(lambda x: ali_utt_index.update({x.split()[0]: x.split()[1:]}), open(next_file_name).readlines())
  except IOError:
    pass
  for i in xrange(1, len(utt_list)):
    try:
      current_file_name, current_line_number = ali_utt_index[utt_list[i]]
      if not (current_file_name == ali_file_name and int(current_line_number) > int(ali_line_number)):
        print current_file_name, ali_file_name, current_line_number, ali_line_number
        commands.append("gunzip -c {1} | tail -n {0}\n".format(int(previous_line_number) - int(ali_line_number) + 1, ali_file_name ))
        ali_file_name = current_file_name
        ali_line_number = current_line_number
    except KeyError:
      continue
    previous_file_name = current_file_name
    previous_line_number = current_line_number
  commands.append("gunzip -c {1} | head -n {0} \n".format(int(previous_line_number) - int(ali_line_number) + 1, ali_file_name ))
  command_file = open(command_filename, "w")
  print "\n".join(commands)
  command_file.write("\n".join(commands))
  command_file.close()
  base, ext = os.path.splitext(command_filename)
  wrapper_file_name = base+'.run'+ext
  command_run_wrapper = open(wrapper_file_name, 'w')
  command_run_wrapper.write("sh " + command_filename + " | gzip -c > " + ali_gz_filename +" \n")
  command_run_wrapper.close()
  print "sh ", wrapper_file_name
  proc = subprocess.Popen(['sh', wrapper_file_name], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  print proc.communicate()
EOF

  # split the ref_data_dir to get reference feats.scp for individual ali.JOB.gz files
  utils/split_data.sh $ref_data_dir $num_jobs
  
  $decode_cmd -v PATH JOB=1:$num_jobs $temp_dir/create_new_ali.JOB.run.log \
    python $temp_dir/create_new_ali.py \
      $ref_data_dir/split$num_jobs/JOB/utt2spk \
      $temp_dir/create_new_ali.JOB.sh $temp_dir/ali.JOB.gz || exit 1;

# check the alignment files generated have at least 98% of the utterances
  for i in `seq 1 $num_jobs`; do
    gunzip -c $temp_dir/ali.$i.gz| awk '{print $1}' > ali_list
    awk '{print $1}' $ref_data_dir/split$num_jobs/$i/utt2spk > feat_list
    num_diff_lines=`diff ali_list feat_list | wc -l`
    num_lines=`cat feat_list|wc -l`
    python -c "import sys;
percent = float($num_diff_lines)/$num_lines*100
if percent > 2 :
  print '$temp_dir/ali.$i.gz {0}% utterances missing.'.format(percent)
  sys.exit(1)"  || exit 1;
  done
  mv  $temp_dir/ali.*.gz $dest
fi
echo "Combined alignments and stored in $dest"
exit 0
