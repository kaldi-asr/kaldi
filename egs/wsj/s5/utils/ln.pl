#!/usr/bin/env perl
use File::Spec;

if ( @ARGV < 2 ) {
  print STDERR "usage: ln.pl input1 input2 dest-dir\n" .
    "This script does a soft link of input1, input2, etc." .
    "to dest-dir, using relative links where possible\n" .
    "Note: input-n and dest-dir may both be absolute pathnames,\n" .
    "or relative pathnames, relative to the current directlory.\n";
  exit(1);
}  

$dir = pop @ARGV;
if ( ! -d $dir ) {
  print STDERR "ln.pl: last argument must be a directory ($dir is not a directory)\n";
  exit(1);
}

$ans = 1; # true.

$absdir = File::Spec->rel2abs($dir); # Get $dir as abs path.
defined $absdir || die "No such directory $dir";
foreach $file (@ARGV) {
  $absfile =  File::Spec->rel2abs($file); # Get $file as abs path.
  defined $absfile || die "No such file or directory: $file";
  @absdir_split = split("/", $absdir);
  @absfile_split = split("/", $absfile);

  $newfile = $absdir . "/" . $absfile_split[$#absfile_split]; # we'll use this
  # as the destination in the link command.
  $num_removed = 0;
  while (@absdir_split > 0 && $absdir_split[0] eq $absfile_split[0]) {
    shift @absdir_split;
    shift @absfile_split;
    $num_removed++;
  }
  if (-l $newfile) { # newfile is already a link -> safe to delete it.
    unlink($newfile); # "unlink" just means delete.
  }
  if ($num_removed == 0) { # will use absolute pathnames.
    $oldfile = "/" . join("/", @absfile_split);
    $ret = symlink($oldfile, $newfile);
  } else {
    $num_dots = @absdir_split;
    $oldfile = join("/", @absfile_split);
    for ($n = 0; $n < $num_dots; $n++) {
      $oldfile = "../" . $oldfile;
    }
    $ret = symlink($oldfile, $newfile);
  }
  $ans = $ans && $ret;
  if (! $ret) {
    print STDERR "Error linking $oldfile to $newfile\n";
  }
}

exit ($ans == 1 ? 0 : 1);

