#!/usr/bin/perl

# Guoguo Chen (guoguo@jhu.edu)
# Not complete yet


if(@ARGV != 1) {
    die "Usage: validate_lang.pl lang_directory\n";
}

$lang = shift @ARGV;

# Checking phones.txt -------------------------------
print "Checking $lang/phones.txt ...\n";
-s "$lang/phones.txt" || die "--> Error: $lang/phones.txt is empty or not exists\n";
open(P, "<$lang/phones.txt") || die "--> Error: fail to open $lang/phones.txt\n";
%psymtab = ();
$idx = 1;
while(<P>) {
    chomp;
    split(" ", $_);
    @_ == 2 || die "--> Error: expect 2 columns in $lang/phones.txt (break at line $idx)\n";
    my $phone = shift @_;
    my $id = shift @_;
    $psymtab{$phone} = $id;
    $idx ++;
}
close(P);
print "--> $lang/phones.txt is OK\n\n";

# Check word.txt -------------------------------
print "Checking words.txt: #0 ...\n";
-s "$lang/words.txt" || die "--> Error: $lang/words.txt is empty or not exists\n";
open(W, "<$lang/words.txt") || die "--> Error: fail to open $lang/words.txt\n";
$idx = 1;
%wsymtab = ();
while(<W>) {
    chomp;
    split(" ", $_);
    @_ == 2 || die "--> Error: expect 2 columns in $lang/words.txt (line $idx)\n";
    $word = shift @_;
    $id = shift @_;
    $wsymtab{$word} = $id;
    $idx ++;
}
if (exists $wsymtab{"#0"}) {
    print "--> $lang/words.txt has \"#0\"\n";
    print "--> $lang/words.txt is OK\n";
} else {die "--> Error: $lang/words.txt doesn't have \"#0\"\n";}
print "\n";
close(W);

# Checking phones/* -------------------------------
sub check_txt_int_csl {
    $cat = shift;
    print "Checking $cat.\{txt, int, csl\} ...\n";
    -s "$cat.txt" || return print "--> Error: $cat.txt is empty or not exists\n";
    -s "$cat.int" || return print "--> Error: $cat.int is empty or not exists\n";
    -s "$cat.csl" || return print "--> Error: $cat.csl is empty or not exists\n";
    open(TXT, "<$cat.txt") || return print "--> Error: fail to open $cat.txt\n";
    open(INT, "<$cat.int") || return print "--> Error: fail to open $cat.int\n";
    open(CSL, "<$cat.csl") || return print "--> Error: fail to open $cat.csl\n";
    
    $idx1 = 0;
    while(<TXT>) {
        chomp;
        split(" ", $_);
        @_ == 1 || return print "--> Error: expect 1 column in $cat.txt (break at line ", $idx1+1, ")\n";
        $entry[$idx1] = shift @_;
        $idx1 ++;
    }
    print "--> $idx1 entry/entries in $cat.txt\n";
    
    $idx2 = 0;
    while(<INT>) {
        chomp;
        split(" ", $_);
        @_ == 1 || return print "--> Error: expect 1 column in $cat.int (break at line ", $idx2+1, ")\n";
        $psymtab{$entry[$idx2]} == shift @_ || return print "--> Error: $cat.int doesn't correspond to $cat.txt (break at line ", $idx2+1, ")\n";
        $idx2 ++;
    }
    $idx1 == $idx2 || return print "--> Error: $cat.int doesn't correspond to $cat.txt (break at line ", $idx2+1, ")\n";
    print "--> $cat.int corresponds to $cat.txt\n";
   
    $idx3 = 0;
    while(<CSL>) {
        chomp;
        split(":", $_);
        @_ == $idx1 || return print "--> Error: expect $idx1 block/blocks in $cat.csl (break at line ", $idx3+1, ")\n";
        foreach(0 .. $idx1-1) {
            $psymtab{$entry[$_]} == @_[$_] || return print "--> Error: $cat.csl doesn't correspond to $cat.txt (break at line ", $idx3+1, ", block ", $_+1, ")\n";
        }
        $idx3 ++;
    }
    $idx3 == 1 || return print "--> Error: expect 1 row in $cat.csl (break at line ", $idx3+1, ")\n";
    print "--> $cat.csl corresponds to $cat.txt\n";

    close(TXT);
    close(INT);
    close(CSL);
    return print "--> $cat.\{txt, int, csl\} are OK\n";
}

sub check_txt_int {
    $cat = shift;
    print "Checking $cat.\{txt, int\} ...\n";
    -s "$cat.txt" || return print "--> Error: $cat.txt is empty or not exists\n";
    -s "$cat.int" || return print "--> Error: $cat.int is empty or not exists\n";
    open(TXT, "<$cat.txt") || return print "--> Error: fail to open $cat.txt\n";
    open(INT, "<$cat.int") || return print "--> Error: fail to open $cat.int\n";
    
    $idx1 = 0;
    while(<TXT>) {
        chomp;
        s/^shared split //g;
        s/ nonword$//g;
        s/ begin$//g;
        s/ end$//g;
        s/ internal$//g;
        s/ singleton$//g;
        $entry[$idx1] = $_;
        $idx1 ++; 
    }
    print "--> $idx1 entry/entries in $cat.txt\n";
    
    $idx2 = 0;
    while(<INT>) {
        chomp;
        s/^shared split //g;
        s/ nonword$//g;
        s/ begin$//g;
        s/ end$//g;
        s/ internal$//g;
        s/ singleton$//g;
        split(" ", $_);
        @set = split(" ", $entry[$idx2]);
        @set == @_ || return print "--> Error: $cat.int doesn't correspond to $cat.txt (break at line ", $idx2+1, ")\n";
        foreach (0 .. @set-1) {
            $psymtab{@set[$_]} == @_[$_] || return print "--> Error: $cat.int doesn't correspond to $cat.txt (break at line ", $idx2+1, ", block " ,$_+1, ")\n"
        }
        $idx2 ++;
    }
    $idx1 == $idx2 || return print "--> Error: $cat.int doesn't correspond to $cat.txt (break at line ", $idx2+1, ")\n";
    print "--> $cat.int corresponds to $cat.txt\n";
   
    close(TXT);
    close(INT);
    return print "--> $cat.\{txt, int\} are OK\n";
}

@list1 = ("context_indep", "disambig", "nonsilence", "silence");
@list2 = ("extra_questions", "roots", "sets");
foreach (@list1) {
    check_txt_int_csl("$lang/phones/$_"); print "\n";
}
foreach (@list2) {
    check_txt_int("$lang/phones/$_"); print "\n";
}
if (-e "$lang/phones/word_boundary.txt") {
    check_txt_int("$lang/phones/word_boundary"); print "\n";
}

# Check disjoint and summation
sub intersect(\%\%) {
    local (*a, *b) = @_;
    @itset = ();
    %itset = ();
    foreach (keys %a) {
        if (exists $b{$_} and !$itset{S_}) {
            push(@itset, $_);
            $itset{$_} = 1;
        }
    }
    return @itset;
}

sub check_disjoint {
    print "Checking disjoint: silence.txt, nosilenct.txt, disambig.txt ...\n";
    open(S, "<$lang/phones/silence.txt")    || return print "--> Error: fail to open $lang/phones/silence.txt\n";
    open(N, "<$lang/phones/nonsilence.txt") || return print "--> Error: fail to open $lang/phones/nonsilence.txt\n";
    open(D, "<$lang/phones/disambig.txt")   || return print "--> Error: fail to open $lang/phones/disambig.txt\n";

    $idx = 1;
    while(<S>) {
        chomp;
        split(" ", $_);
        $phone = shift @_;
        ! exists $silence{$phone} || print "--> Error: more than one \"$phone\" exist in $lang/phones/silence.txt (line $idx)\n";
        $silence{$phone} = 1;
        push(@silence, $phone);
        $idx ++;
    }

    $idx = 1; 
    while(<N>) {
        chomp;
        split(" ", $_);
        $phone = shift @_;
        ! exists $nonsilence{$phone} || print "--> Error: more than one \"$phone\" exist in $lang/phones/nonsilence.txt (line $idx)\n";
        $nonsilence{$phone} = 1;
        push(@nonsilence, $phone);
        $idx ++;
    }

    $idx = 1;
    while(<D>) {
        chomp;
        split(" ", $_);
        $phone = shift @_;
        ! exists $disambig{$phone} || print "--> Error: more than one \"$phone\" exist in $lang/phones/disambig.txt (line $idx)\n";
        $disambig{$phone} = 1;
        $idx ++;
    }

    my @itsect1 = intersect(%silence, %nonsilence);
    my @itsect2 = intersect(%silence, %disambig);
    my @itsect3 = intersect(%disambig, %nonsilence);

    $success = 1;
    if (@itsect1 != 0) {
        $success = 0;
        print "--> Error: silence.txt and nonsilence.txt have intersection -- ";
        foreach (@itsect1) {
            print $_, " ";
        }
        print "\n";
    } else {print "--> silence.txt and nonsilence.txt are disjoint\n";}
    
    if (@itsect2 != 0) {
        $success = 0;
        print "--> Error: silence.txt and disambig.txt have intersection -- ";
        foreach (@itsect2) {
            print $_, " ";
        }
        print "\n";
    } else {print "--> silence.txt and disambig.txt are disjoint\n";}

    if (@itsect3 != 0) {
        $success = 0;
        print "--> Error: disambig.txt and nonsilence.txt have intersection -- ";
        foreach (@itsect1) {
            print $_, " ";
        }
        print "\n";
    } else {print "--> disambig.txt and nonsilence.txt are disjoint\n";}
    
    close(S);
    close(N);
    close(D);
    $success == 0 || print "--> disjoint property is OK\n";
    return;
}

sub check_summation {
    print "Checking sumation: silence.txt, nonsilence.txt, disambig.txt ...\n";
    scalar(keys %silence) != 0      || return print "--> Error: $lang/phones/silence.txt is empty or not exists\n";
    scalar(keys %nonsilence) != 0   || return print "--> Error: $lang/phones/nonsilence.txt is empty or not exists\n";
    scalar(keys %disambig) != 0     || return print "--> Error: $lang/phones/disambig.txt is empty or not exists\n";

    %sum = (%silence, %nonsilence, %disambig);
    $sum{"<eps>"} = 1;

    my @itset = intersect(%sum, %psymtab);
    my @key1 = keys %sum;
    my @key2 = keys %psymtab;
    my %itset = (); foreach (@itset) {$itset{$_} = 1;}
    if (@itset < @key1) {
        print "--> Error: phones in silence.txt, nonsilence.txt, disambig.txt but not in phones.txt -- ";
        foreach (@key1) {
            if (!$itset{$_}) {print "$_ ";}
        }
        print "\n";
    }

    if (@itset < @key2) {
        print "--> Error: phones in phones.txt but not in silence.txt, nonsilence.txt, disambig.txt -- ";
        foreach (@key2) {
            if (!$itset{$_}) {print "$_ ";}
        }
        print "\n";
    }

    if (@itset == @key1 and @itset == @key2) {
        print "--> summation property is OK\n";
    }
    return;
}

%silence = ();
@silence = ();
%nonsilence = ();
@nonsilence = ();
%disambig = ();
check_disjoint; print "\n";
check_summation; print "\n";

# Check disambiguation symbols
print "Checking disambiguation symbols: #0 and #1\n";
scalar(keys %disambig) != 0 || print "--> Error: $lang/phones/disambig.txt is empty or not exists\n";
if (exists $disambig{"#0"} and exists $disambig{"#1"}) {
    print "--> $lang/phones/disambig.txt has \"#0\" and \"#1\"\n";
    print "--> $lang/phones/disambig.txt is OK\n\n";
} else {
    print "--> Error: $lang/phones/disambig.txt doesn't have \"#0\" or \"#1\"\n";
}


# Check topo
print "Checking topo ...\n";
-s "$lang/topo" || print "--> Error: $lang/topo is empty or not exists\n";
open(T, "<$lang/topo") || print "--> Error: fail to open $lang/topo\n";
$idx = 1;
while(<T>) {
    chomp;
    next if (m/^<.*>[ ]*$/);
    if ($idx == 1) {$nonsilence_seq = $_; $idx ++;}
    if ($idx == 2) {$silence_seq = $_;}
}
$silence_seq != 0 and $nonsilence_seq != 0 || print "--> Error: $lang/topo doesn't have nonsilence section or silence section\n";
@silence_seq = split(" ", $silence_seq);
@nonsilence_seq = split(" ", $nonsilence_seq);
$success1 = 1;
if (@nonsilence_seq != @nonsilence) {print "--> Error: $lang/topo's nonsilence section doesn't correspond to nonsilence.txt\n";}
else {
    foreach (0 .. scalar(@nonsilence)-1) {
        if ($psymtab{@nonsilence[$_]} != @nonsilence_seq[$_]) {
            print "--> Error: $lang/topo's nonsilence section doesn't correspond to nonsilence.txt\n";
            $success = 0;
        }
    }
}
$success1 != 1 || print "--> $lang/topo's nonsilence section is OK\n";
$success2 = 1;
if (@silence_seq != @silence) {print "--> Error: $lang/topo's silence section doesn't correspond to silence.txt\n";}
else {
    foreach (0 .. scalar(@silence)-1) {
        if ($psymtab{@silence[$_]} != @silence_seq[$_]) {
            print "--> Error: $lang/topo's silence section doesn't correspond to silence.txt\n";
            $success = 0;
        }
    }
}
$success2 != 1 || print "--> $lang/topo's silence section is OK\n";
$success1 != 1 or $success2 != 1 || print "--> $lang/topo is OK\n";
print "\n";
close(T);

# Check word_boundary
if (-s "$lang/phones/word_boundary.txt") {
    print "Checking word_boundary.txt: silence.txt, nonsilence.txt, disambig.txt ...\n";
    open (W, "<$lang/phones/word_boundary.txt") || print "--> Error: fail to open $lang/phones/word_boundary.txt\n";
    $idx = 1;
    %wb = ();
    while(<W>) {
        chomp;
        s/ nonword$//g;
        s/ begin$//g;
        s/ end$//g;
        s/ internal$//g;
        s/ singleton$//g;
        split(" ", $_);
        @_ == 1 || print "--> Error: expect 1 column in $lang/phones/word_boundary.txt (line $idx)\n";
        $wb{shift @_} = 1;
        $idx ++;
    }
    close(W);

    @itset = intersect(%disambig, %wb);
    $success1 = 1;
    if (@itset != 0) {
        $success1 = 0;
        print "--> Error: $lang/phones/word_boundary.txt has disambiguation symbols -- ";
        foreach (@itset) {print "$_ ";}
        print "\n";
    }
    $success1 == 0 || print "--> $lang/phones/word_boundary.txt doesn't include disambiguation symbols\n";

    %sum = (%silence, %nonsilence);
    @itset = intersect(%sum, %wb);
    %itset = (); foreach (@itset) {$itset{$_} = 1;}
    $success2 = 1;
    if (@itset < scalar(keys %sum)) {
        $success2 = 0;
        print "--> Error: phones in nonsilence.txt and silence.txt but not in word_boundary.txt -- ";
        foreach (keys %sum) {
            if (!$itset{$_}) {print "$_ ";}            
        }
        print "\n";
    }
    if (@itset < scalar(keys %wb)) {
        $success2 = 0;
        print "--> Error: phones in word_boundary.txt but not in nonsilence.txt or silence.txt -- ";
        foreach (keys %wb) {
            if (!$itset{$_}) {print "$_ ";}
        }
        print "\n";
    }
    $success2 == 0 || print "--> $lang/phones/word_boundary.txt is the union of nonsilence.txt and silence.txt\n";
    $success1 != 1 or $success2 != 1 || print "--> $lang/phones/word_boundary.txt is OK\n"; print "\n";
}

