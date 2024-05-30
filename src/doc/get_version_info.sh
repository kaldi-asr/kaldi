#!/usr/bin/env bash

# search for VERSION below to see how to change this when
# Kaldi's version number increases.

# Note: this script assumes that it's part of a git repository where
# the official kaldi repo is a remote named 'upstream', as shown
# here:
# git remote -vv | grep upstream
# upstream	git@github.com:kaldi-asr/kaldi.git (fetch)
# upstream	git@github.com:kaldi-asr/kaldi.git (push)
# Since Dan is going to be the one running this script and that's
# how he does it, this should work fine.



# the tuples are:   <major/minor number> <branch on github> <first commit of that version>

if [ "$0" != "doc/get_version_info.sh" ] || [ $# -ne 0 ]; then
  echo "$0: you should run this script without arguments, from the src/ directory."
  echo "... It generates 5.0.html, 5.1.html, and so on."
fi

if ! git fetch upstream; then
  echo "$0: command 'git fetch upstream' failed"
  exit 1
fi


# echo "fooXXabcYYbar" | perl -ane ' if (m/XX(.+)YY/) { $a=$`;$x=$1;$y=$'\''; $x =~ s/a/b/g; print "${a}XX${x}YY${y}"; } else {print;}'

# VERSION
# When you increment the version of Kaldi you'll want to
# add a new tuple to the 'for' statement, of the form "5.x master yyyy".
# where the tuples will generally be of the form: "x.x master yyyyyy"
# where yyyyy is the result of git log -1 src/.version on
# that version of Kaldi (we only update the .version file when
# the major/minor version number changes).
# You should change the previous 'master' to the previous version number,
# e.g. 5.2 (this assumes you created a branch in Kaldi's repository that
# archives that version of Kaldi).
# Note: when you add new tuples here you'll also want to add ndew
# \htmlinclude directives in versions.dox.

for tuple in "5.0 5.0 c160a9883" "5.1 5.1 2145519961" "5.2 5.2 393ef73caa93" "5.3 5.3 db28650346ba07" \
                                 "5.4 5.4 be969d7baf04" "5.5 master 7aab92b7c"; do
  if [ $(echo $tuple | wc -w) != 3 ]; then
    echo "$0: tuple should have 3 fields: '$tuple'"
    exit 1
  fi
  major_minor_number=$(echo $tuple | awk '{print $1}')  # e.g. 5.0
  branch=$(echo $tuple | awk '{print $2}')  # e.g. 'master', or '5.1' (it's a branch name)
  first_commit=$(echo $tuple | awk '{print $3}')


  tempfile=$(mktemp /tmp/temp.XXXXXX)
  echo "$0: for version=$major_minor_number, writing git output to $tempfile"

  patch_number=0
  # git rev-list --reverse $first_commit..upstream/$branch  lists the revisions from
  # $first_commit to upstream/$branch... --boundary causes it to include $first_commit
  # in the range, but with a dash (-) included for the first commit, so we
  # use a sed command to get rid of that.
  for rev in $(git rev-list --reverse $first_commit..upstream/$branch --boundary | sed s/-//); do
    # %h is abbrev. commit hash, %H is long commit hash, %cd is the commit date,
    # %%s is the one-line log message; x09 is tab.
    # so we're printing "<patch-number> <short-commit> <long-commit> <commit-date> <commit-subject>"
    # we'll later parse this and generate HTML.
    pretty_str="${patch_number}%x09%h%x09%H%x09%cd%x09%s";
    git log --date=short --pretty="$pretty_str" -1 $rev
    patch_number=$[patch_number+1]
  done > $tempfile

  htmlfile=doc/$major_minor_number.html
  echo "$0: for version=$major_minor_number, processing $tempfile to $htmlfile"

  cat $tempfile | perl -e '
    ($major_minor_number) = @ARGV;
    while (<STDIN>) {
      if (! m/^(\S+)\t(\S+)\t(\S+)\t(\S+)\t(.+)/) {
         die "Could not parse line $_ in git output";
      } else {
        $patch_number = $1; $short_commit = $2; $long_commit = $3;
        $commit_date = $4; $commit_subject = $5;
        if ($commit_subject =~ m/\(#(\d+)\)\s*$/) {
           $pull_request_number = $1;
           $pre_match = $`;  # part before what was matched.
           $pre_match =~ s/</&lt;/g;
           $pre_match =~ s/>/&gt;/g;
           # if commit subject line ends with e.g. (#1302), which will
           # be a pull request; create a href to github for that.
           $commit_subject = $pre_match .
            "<a href=\"https://github.com/kaldi-asr/kaldi/pull/$pull_request_number\" target=\"_blank\">(#$pull_request_number)</a>";
         } else {
           $commit_subject =~ s/</&lt;/g;
           $commit_subject =~ s/>/&gt;/g;
         }
         $commit_href =
          "<a href=\"https://github.com/kaldi-asr/kaldi/commit/$long_commit\" target=\"_blank\">$short_commit</a>";
         $line = "$major_minor_number.$patch_number $commit_href $commit_date $commit_subject <br>\n";
         print $line;
      }
      print "<p>\n";
    } ' "$major_minor_number" >$htmlfile || exit 1
  echo "$0: generated file $htmlfile with $(wc -l <$htmlfile) lines"
  # you might want to comment the command below if you are debugging the script.
  rm $tempfile
done
