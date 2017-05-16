#!/usr/bin/perl

my $os=`uname | tr -d '\012'`;
my $dir=`dirname $0 | tr -d '\012'`;
my $astyle="$dir/astyle_$os";

opendir(DIR,".") or die "Can't open the current directory: $!\n";

# read file/directory names in that directory into @names 
@names = readdir(DIR) or die "Unable to read current dir:$!\n";

foreach $name (@names) {
   next if ($name eq ".");   # skip the current directory entry
   next if ($name eq "..");  # skip the parent  directory entry

   if (-d $name){            # is this a directory?
      `$astyle --style="k&r" -s2 --recursive -v "$name/*.h" "$name/*.cpp"`;
      next;                  # can skip to the next name in the for loop 
   }
}

closedir(DIR);
