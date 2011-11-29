#!/usr/bin/perl

# In general, doing run.pl some.log a b c is like running the command a b c in
# the bash shell, and putting the standard error and output into some.log. 
# A typical example is:
#  run.pl some.log my-prog "--opt=foo bar" foo \|  other-prog baz
# and run.pl will run something like:
# ( my-prog '--opt=foo bar' foo |  other-prog baz ) >& some.log
# 
# Basically it takes the command-line arguments, quotes them
# as necessary to preserve spaces, and evaluates them with bash.
# In addition it puts the command line at the top of the log, and 
# the start and end times of the command at the beginning and end.
# The reason why this is useful is so that we can create a different
# version of this program that uses a queueing system instead.

@ARGV < 2 && die "usage: run.pl log-file command-line arguments...";
$logfile = shift @ARGV;

$cmd = "";

foreach $x (@ARGV) { 
    if ($x =~ m/^\S+$/) { $cmd .=  $x . " "; }
    elsif ($x =~ m:\":) { $cmd .= "'\''$x'\'' "; }
    else { $cmd .= "\"$x\" "; } 
}


open(F, ">$logfile") || die "Error opening log file $logfile";
print F "# " . $cmd . "\n";
print F "# Started at " . `date`;
$starttime = `date +'%s'`;
print F "#\n";
close(F);

# Pipe into bash.. make sure we're not using any other shell.
open(B, "|bash") || die "Error opening shell command"; 
print B "( " . $cmd . ") 2>>$logfile >> $logfile";
close(B); # If there was an error, exit status is in $?
$ret = $?;

$endtime = `date +'%s'`;
open(F, ">>$logfile") || die "Error opening log file $logfile (again)";
$enddate = `date`;
chop $enddate;
print F "# Ended (code $ret) at " . $enddate . ", elapsed time " . ($endtime-$starttime) . " seconds\n";
close(F);
exit($ret == 0 ? 0 : 1);
