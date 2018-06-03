#!/usr/bin/env perl
#
# Copyright (c) 1994-1996 Wolfram Schneider <wosch@FreeBSD.org>. Berlin.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# makewhatis -- update the whatis database in the man directories.
#
# $FreeBSD: src/gnu/usr.bin/man/makewhatis/makewhatis.perl,v 1.21.2.4 2002/03/29 15:38:09 ru Exp $


sub usage {

    warn <<EOF;
usage: makewhatis [-a|-append] [-h|-help] [-i|-indent column] [-L|-locale]
                  [-n|-name name] [-o|-outfile file] [-v|-verbose]
                  [directories ...]
EOF
    exit 1;
}


# Format output
sub open_output {
    local($dir) = @_;

    die "Name for whatis is empty\n" if $whatis_name eq "";

    if ($outfile) {		# Write all Output to $outfile
	$whatisdb = $outfile;
    } else {		# Use man/whatis
	$whatisdb = $dir . "/$whatis_name.tmp";
    }
    $tmp = $whatisdb;		# for signals

    # Array of all entries
    @a = ();

    # Append mode
    if ($append) {
	local($file) = $whatisdb;
	$file =~ s/\.tmp$// if !$outfile;
	
	if (open(A, "$file")) {
	    warn "Open $file for append mode\n" if $verbose;	    
	    while(<A>) {
		push(@a, $_);
	    }
	    close A;
	} 

	else {
	    warn "$whatisdb: $!\n" if lstat($file) && $verbose;	# 
	}
	undef $file;
    }


    warn "Open $whatisdb\n" if $verbose;
    if (!open(A, "> $whatisdb")) {
        die "$whatisdb: $!\n" if $outfile;

        warn "$whatisdb: $!\n"; $err++; return 0;
    }
 
    select A;
    return 1;
}

sub close_output {
    local($success) = @_;
    local($w) = $whatisdb;
    local($counter) = 0;
    local($i, $last,@b);

    $w =~ s/\.tmp$//;
    if ($success) {		# success
	# uniq
	warn "\n" if $verbose && $pointflag;
	warn "sort -u > $whatisdb\n" if $verbose;
	foreach $i (sort @a) {
	    if ($i ne $last) {
		push(@b, $i);
	    }
	    $last =$i;
	}

	$counter = $#b + 1;
	print @b; close A; select STDOUT;

	if (!$outfile) {
	    warn "Rename $whatisdb to $w\n" if $verbose;
	    rename($whatisdb, $w) || warn "rename $whatisdb $w\n";
	    $counter_all += $counter;
	    warn "$counter entries in $w\n" if $verbose;
	} else {
	    $counter_all = $counter;
	}
    } else {		# building whatisdb failed
	unlink($whatisdb);
	warn "building whatisdb: $whatisdb failed\n" if $verbose;
    }
    return 1;
}

sub parse_subdir {
    local($dir) = @_;
    local($file, $dev,$ino);

    warn "\n" if $pointflag;
    warn "traverse $dir\n" if $verbose;
    $pointflag = 0;

    if (!opendir(M, $dir)) {
	warn "$dir: $!\n"; $err++; return 0;
    }

    $| = 1 if $verbose;
    foreach $file (readdir(M)) {
	next if $file =~ /^(\.|\.\.)$/;

	($dev, $ino) = ((stat("$dir/$file"))[01]);
	if (-f _) {
	    if ($man_red{"$dev.$ino"}) {
		# Link
		print STDERR "+" if $verbose;
		$pointflag++ if $verbose;
	    } else {
		&manual("$dir/$file");
	    }
	    $man_red{"$dev.$ino"} = 1;
	} elsif (! -d _) {
	    warn "Cannot find file: $dir/$file\n"; $err++;
	}
    }
    closedir M;
    return 1;
}

# read man directory
sub parse_dir {
    local($dir) = @_;
    local($subdir, $file);

    # clean up, in case mandir and subdirs are called simultaneously
    # e. g.:  ~/man/man1 ~/man/man2 ~/man
    #~/man/ man1 and ~/man/man2 are a subset of ~/man
    foreach $file (keys %man_red) {
	delete $man_red{$file};
    }

    if ($dir =~ /man/) {
	warn "\n" if $verbose && $pointflag;
	warn "open manpath directory ``$dir''\n" if $verbose;
	$pointflag = 0;
	if (!opendir(DIR, $dir)) {
	    warn "opendir ``$dir'':$!\n"; $err = 1; return 0;
	}
	foreach $subdir (sort(readdir(DIR))) {
	    if ($subdir =~ /^man\w+$/) {
		$subdir = "$dir/$subdir";
		&parse_subdir($subdir);
		&parse_subdir($subdir) if -d ($subdir .= "/${machine}");
	    }
	}
	closedir DIR

    } elsif ($dir =~ /man\w+$/) {
	&parse_subdir($dir);
    } else {
	warn "Assume ``$dir'' is not a man directory.\n";
	$err = 1; return 0;
    }
    return 1;
}

sub dir_redundant {
    local($dir) = @_;

    local($dev,$ino) = (stat($dir))[0..1];

    if ($dir_redundant{"$dev.$ino"}) {
	warn "$dir is equal to: $dir_redundant{\"$dev.$ino\"}\n" if $verbose;
	return 0;
    }
    $dir_redundant{"$dev.$ino"} = $dir;
    return 1;
}


# ``/usr/man/man1/foo.l'' -> ``l''
sub ext {
    local($filename) = @_;
    local($extension) = $filename;

    $extension =~ s/$ext$//g;	# strip .gz
    $extension =~ s/.*\///g;	# basename

    if ($extension !~ m%[^/]+\.[^.]+$%) {	# no dot
	$extension = $filename;
	#$extension =~ s|/[^/]+$||;
	$extension =~ s%.*man([^/]+)/[^/]+%$1%; # last character
	warn "\n" if $verbose && $pointflag;
	warn "$filename has no extension, try section ``$extension''\n"
	    if $verbose;
	$pointflag = 0;
    } else {
	$extension =~ s/.*\.//g; # foo.bla.1 -> 1
    }
    return "$extension";
}

# ``/usr/man/man1/foo.1'' -> ``foo''
sub name {
    local($name) = @_;

    $name =~ s=.*/==;
    $name =~ s=$ext$==o;
    $name =~ s=\.[^\.]+$==;

    return "$name";
}

# output
sub out {
    local($list) = @_;
    local($delim) = " - ";
    $_ = $list;

    # delete italic etc.
    s/^\.[^ -]+[ -]+//;
    s/\\\((em|mi)//;
    s/\\f[IRBP]//g;
    s/\\\*p//g;
    s/\(OBSOLETED\)[ ]?//;
    s/\\&//g;
    s/^\@INDOT\@//;
    s/[\"\\]//g;		#"
    s/[. \t-]+$//;

    s/ / - / unless / - /;
    ($man,$desc) = split(/ - /);

    $man = $name unless $man;
    $man =~ s/[,. ]+$//;
    $man =~ s/,/($extension),/g;
    $man .= "($extension)";

    &manpagename;

    $desc =~ s/^[ \t]+//;

    for($i = length($man); $i < $indent && $desc; $i++) {
	$man .= ' ';
    }
    if ($desc) {
	push(@a, "$man$delim$desc\n");
    } else {
	push(@a, "$man\n");
    }
}

# The filename of manual page is not a keyword. 
# This is bad, because you don't find the manpage
# whith: $ man <section> <keyword>
#
# Add filename if a) filename is not a keyword and b) no keyword(s)
# exist as file in same mansection
#
sub manpagename {
    foreach (split(/,\s+/, $man)) {
	s/\(.+//;
	# filename is keyword
	return if $name eq $_;
    }

    local($f) = $file;  $f =~ s%/*[^/]+$%%;		# dirname
    local($e) = $file;  $e =~ s/$ext$//;  $e =~ s%.*(\.[^.]+)$%$1%; # .1

    foreach (split(/,\s+/, $man)) {
	s/\(.+//;

	# a keyword exist as file
	return if -e "$f/$_$e" || -e "$f/$_$e$ext";    
    }

    $man = "$name($extension), $man";
}

# looking for NAME
sub manual {
    local($file) = @_;
    local($list, $desc, $extension);
    local($ofile) = $file;

    # Compressed man pages
    if ($ofile =~ /$ext$/) {
	$ofile = "gzcat $file |";
	print STDERR "*" if $verbose;
    } else {
	print STDERR "." if $verbose;
    }
    $pointflag++ if $verbose;

    if (!open(F, "$ofile")) {
	warn "Cannot open file: $ofile\n"; $err++;
	return 0;
    }
    # extension/section
    $extension = &ext($file);
    $name = &name($file);

    $section_name = "NAME|Name|NAMN|BEZEICHNUNG|ּ¾¾־|מבתקבמיו";

    local($source) = 0;
    local($list);
    while(<F>) {
	# ``man'' style pages
	# &&: it takes you only half the user time, regexp is slow!!!
 	if (/^\.SH/ && /^\.SH[ \t]+["]?($section_name)["]?/) {
	    #while(<F>) { last unless /^\./ } # Skip
	    #chop; $list = $_;
	    while(<F>) {
		last if /^\.SH[ \t]/;
		chop;
		s/^\.IX\s.*//;            # delete perlpod garbage
		s/^\.[A-Z]+[ ]+[0-9]+$//; # delete commands
		s/^\.[A-Za-z]*[ \t]*//;	  # delete commands
		s/^\.\\".*$//;            #" delete comments
		s/^[ \t]+//;
		if ($_) {
		    $list .= $_;
		    $list .= ' ';
		}
	    }
	    while(<F>) { }	# skip remaining input to avoid pipe errors
	    &out($list); close F; return 1;
 	} elsif (/^\.Sh/ && /^\.Sh[ \t]+["]?($section_name)["]?/) {
	    # ``doc'' style pages
	    local($flag) = 0;
	    while(<F>) {
		last if /^\.Sh/;
		chop;
		s/^\.\\".*$//;            #" delete comments
		next if /^\.[ \t]*$/;	  # skip empty calls
		if (/^\.Nm/) {
		    s/^\.Nm[ \t]*//;
		    s/ ,/,/g;
		    s/[ \t]+$//;
		    $list .= $_;
		    $list .= ' ';
		} else {
		    $list .= '- ' if (!$flag && !/^- /);
		    $flag++;
		    if (/^\.Xr/) {
			split;
			$list .= @_[1];
			$list .= "(@_[2])" if @_[2];
		    } else {
			s/^\.([A-Z][a-z])?[ \t]*//;
			s/[ \t]+$//;
			$list .= $_;
		    }
		    $list .= ' ';
		}
	    }
	    while(<F>) { }	# skip remaining input to avoid pipe errors
	    &out($list); close F; return 1;

	} elsif(/^\.so/ && /^\.so[ \t]+man/) {
	    while(<F>) { }	# skip remaining input to avoid pipe errors
	    close F; return 1;
	}
    }
    if (!$source && $verbose) {
	warn "\n" if $pointflag;
	warn "Maybe $file is not a manpage\n" ;
	$pointflag = 0;
    }
    return 0;
}

# make relative path to absolute path
sub absolute_path {
    local(@dirlist) = @_;
    local($pwd, $dir, @a);

    $pwd = $ENV{'PWD'};
    foreach $dir (@dirlist) {
	if ($dir !~ "^/") {
	    chop($pwd = `pwd`) if (!$pwd || $pwd !~ /^\//);
	    push(@a, "$pwd/$dir");
	} else {
	    push(@a, $dir);
	}
    }
    return @a;
}

# strip unused '/'
# e.g.: //usr///home// -> /usr/home
sub stripdir {
    local($dir) = @_;

    $dir =~ s|/+|/|g;		# delete double '/'
    $dir =~ s|/$||;		# delete '/' at end
    $dir =~ s|/(\.\/)+|/|g;	# delete ././././

    $dir =~ s|/+|/|g;		# delete double '/'
    $dir =~ s|/$||;		# delete '/' at end
    $dir =~ s|/\.$||;		# delete /. at end
    return $dir if $dir ne "";
    return '/';
}

sub variables {
    $verbose = 0;		# Verbose
    $indent = 24;		# Indent for description
    $outfile = 0;		# Don't write to ./whatis
    $whatis_name = "whatis.db";	# Default name for DB
    $append = 0;		# Don't delete old entries
    $locale = 0;		# Build DB only for localized man directories
    chomp($machine = $ENV{'MACHINE'} || `uname -m`);

    # choose localized man directories suffix.
    $local_suffix = $ENV{'LC_ALL'} || $ENV{'LC_CTYPE'} || $ENV{'LANG'};

    # if no argument for directories given
    @defaultmanpath = ( '/usr/share/man' );

    $ext = '.gz';		# extension
    umask(022);

    $err = 0;			# exit code
    $whatisdb = '';
    $counter_all = 0;
    $dir_redundant = '';	# redundant directories
    $man_red = '';		# redundant man pages
    @a = ();			# Array for output

    # Signals
    $SIG{'INT'} = 'Exit';
    $SIG{'HUP'} = 'Exit';
    $SIG{'TRAP'} = 'Exit';
    $SIG{'QUIT'} = 'Exit';
    $SIG{'TERM'} = 'Exit';
    $tmp = '';			# tmp file

    $ENV{'PATH'} = "/bin:/usr/bin:$ENV{'PATH'}";
}

sub  Exit {
    unlink($tmp) if $tmp ne ""; # unlink if a filename
    die "$0: die on signal SIG@_\n";
}

sub parse {
    local(@argv) = @_;
    local($i);

    while ($_ = $argv[0], /^-/) {
	shift @argv;
	last if /^--$/;
	if    (/^--?(v|verbose)$/)      { $verbose = 1 }
	elsif (/^--?(h|help|\?)$/)      { &usage }
	elsif (/^--?(o|outfile)$/)      { $outfile = $argv[0]; shift @argv }
	elsif (/^--?(f|format|i|indent)$/) { $i = $argv[0]; shift @argv }
	elsif (/^--?(n|name)$/)         { $whatis_name = $argv[0];shift @argv }
	elsif (/^--?(a|append)$/)       { $append = 1 }
	elsif (/^--?(L|locale)$/)       { $locale = 1 }
	else                            { &usage }
    }
    warn "Localized man directory suffix is ``$local_suffix''\n"
	if $verbose && $locale;

    if ($i ne "") {
	if ($i =~ /^[0-9]+$/) {
	    $indent = $i;
	} else {
	    warn "Ignoring wrong indent value: ``$i''\n";
	}
    }

    return &absolute_path(@argv) if $#argv >= 0;
    return @defaultmanpath if $#defaultmanpath >= 0;

    warn "Missing directories\n"; &usage;
}

# Process man directory
sub process_dir {
  local($dir) = @_;

  $dir = &stripdir($dir);
  &dir_redundant($dir) && &parse_dir($dir);
}

# Process man directory and store output to file
sub process_dir_to_file {
  local($dir) = @_;

  $dir = &stripdir($dir);
  &dir_redundant($dir) &&
      &close_output(&open_output($dir) && &parse_dir($dir));
} 

# convert locale name to short notation (ru_RU.KOI8-R -> ru.KOI8-R)
sub short_locale_name {
  local($lname) = @_;

  $lname =~ s|_[A-Z][A-Z]||;
  warn "short locale name is $lname\n" if $verbose && $locale;
  return $lname;
}

##
## Main
##

&variables;
@argv = &parse( @ARGV );

if ($outfile) {
    if(&open_output($outfile)){
	foreach $dir (@argv) {
	    # "Local only" flag set ? Yes ...
	    if ($locale) {
		if ($local_suffix ne "") {
		     &process_dir($dir.'/'.$local_suffix);
		     &process_dir($dir.'/'.&short_locale_name($local_suffix));
		}
	    } else {
		&process_dir($dir);
	    }
	}
    }
    &close_output(1);
} else {
    foreach $dir (@argv) {
	# "Local only" flag set ? Yes ...
        if ($locale) {
	    if ($local_suffix ne "") {
	      &process_dir_to_file($dir.'/'.$local_suffix);
	      &process_dir_to_file($dir.'/'.&short_locale_name($local_suffix));
	    }
	} else {
	   &process_dir_to_file($dir);
	}
    }
}

warn "Total entries: $counter_all\n" if $verbose && ($#argv > 0 || $outfile);
exit $err;
