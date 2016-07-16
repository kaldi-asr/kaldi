#!/usr/bin/env bash

# run this from ../..

if [ "$1" == "--really" ]; then
  really_do_it=true
else
  really_do_it=false
  echo "$0: this will not really do anything, use --really for that."
fi


cd src

cat Makefile | perl -e ' @libdirs = ();  while(<>){ if ($x && m/(\S+)\:/) { push @libdirs, $1; chop; print "$_ .phony\n"; } if (m/have inter-dependencies/) {$x=1;}} print("all: " . join(" ", @libdirs) . "\n"); print(".phony:\n")' > Makefile.temp

# for each directory this automatic rule says, just print its name.
echo '%:' >> Makefile.temp
printf '\techo ${@F}\n' >> Makefile.temp

# the following prints out the directory names in the order we want to
# have them in the ADDLIBS in the individual Makefiles; note, 'tac' reverses
# the order of its input lines.
make -s -f Makefile.temp all | grep -v all | grep -v .phony | tac > library_order

echo "Library order is:"
cat library_order

for f in */Makefile; do
  echo "$0: processing $f"
  cat $f | perl -e '
    open(F, "<library_order") || die "opening file library_order";
    $n = 1;
    while (<F>) { chop; $library_name_to_order{$_} = $n; $order_to_library_name{$n} = $_; $n++; }
    while(<>) {
      if (m/^ADDLIBS = (.+)/) {
        @addlibs = ();
        $cur_line = $1;
        while (1) {
           if ($cur_line =~ s/\\$//) { $had_backslash = 1; } else { $had_backslash = 0; }
           @A = split(" ", $cur_line);
           push @addlibs, @A;
           if (!$had_backslash) { last; }  # break from the while loop.
           if (!($cur_line = <>)) { last; }
        }
        @weird_libs = ();
        %normal_lib_names = {};
        foreach $lib (@addlibs) {
          if ($lib =~ m|^\.\./(.+)/kaldi-(.+)\.a$| && $1 == $2 && defined $library_name_to_order{$1}) {
             $normal_lib_names{$1} = 1;
          } else { push @weird_libs, $lib; }
        }
        @normalized_addlibs = ();
        for ($k = 1; $k < $n; $k++) {
           $test_name = $order_to_library_name{$k};
            if (defined $normal_lib_names{$test_name}) {
               push @normalized_addlibs, "../$test_name/kaldi-$test_name.a";
            }
         }
        if (@weird_libs > 0) {  print STDERR "Unexpected libraries: " . join(":", @weird_libs) . "\n"; }
        # unexpected libraries that aren not part of the normal list will go last.
        push @normalized_addlibs, @weird_libs;
        @rearranged_lines = ();
        $cur_line = "";
        $max_partial_line_size = 70; # after the initial "ADDLIBS = " or spaces.
        foreach $lib (@normalized_addlibs) {
            if (length($cur_line . $lib . " ") > $max_partial_line_size) {
                push @rearranged_lines, $cur_line; $cur_line = "";
            }
            $cur_line .= ($lib . " ");
        }
        if ($cur_line ne "") { push @rearranged_lines, $cur_line; }
        $num_lines = @rearranged_lines;
        for ($k = 0; $k < $num_lines; $k++) {
           if ($k == 0) { print "ADDLIBS = "; } else { print "          "; }
           print $rearranged_lines[$k];
           if ($k + 1 < $num_lines) { print "\\\n"; } else { print "\n"; }
        }
      } else {
        print;
      }
    } ' > temp_makefile
   diff $f temp_makefile
   if $really_do_it; then
     cp temp_makefile $f
   fi
done

rm library_order Makefile.temp
