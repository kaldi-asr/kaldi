#!/usr/bin/perl

# Guoguo Chen (guoguo@jhu.edu)
#
# Validation script for data/local/dict

if(@ARGV != 1) {
    die "Usage: validate_dict_dir.pl dict_directory\n";
}

$dict = shift @ARGV;

$exit = 0;
# Checking silence_phones.txt -------------------------------
print "Checking $dict/silence_phones.txt ...\n";
if(-z "$dict/silence_phones.txt") {print "--> ERROR: $dict/silence_phones.txt is empty or not exists\n"; exit 1;}
if(!open(S, "<$dict/silence_phones.txt")) {print "--> ERROR: fail to open $dict/silence_phones.txt\n"; exit 1;}
$idx = 1;
%silence = ();
$success = 1;
print "--> reading $dict/silence_phones.txt\n";
while(<S>) {
    chomp;
    my @col = split(" ", $_);
    foreach(0 .. @col-1) {
        if($silence{@col[$_]}) {$exit = 1; print "--> ERROR: phone \"@col[$_]\" duplicates in $dict/silence_phones.txt (line $idx)\n"; $success = 0;}
        else {$silence{@col[$_]} = 1;}
    }
    $idx ++;
}
close(S);
$success == 0 || print "--> $dict/silence_phones.txt is OK\n";
print "\n";

# Checking nonsilence_phones.txt -------------------------------
print "Checking $dict/nonsilence_phones.txt ...\n";
if(-z "$dict/nonsilence_phones.txt") {print "--> ERROR: $dict/nonsilence_phones.txt is empty or not exists\n"; exit 1;}
if(!open(NS, "<$dict/nonsilence_phones.txt")) {print "--> ERROR: fail to open $dict/nonsilence_phones.txt\n"; exit 1;}
$idx = 1;
%nonsilence = ();
$success = 1;
print "--> reading $dict/nonsilence_phones.txt\n";
while(<NS>) {
    chomp;
    my @col = split(" ", $_);
    foreach(0 .. @col-1) {
        if($nonsilence{@col[$_]}) {$exit = 1; print "--> ERROR: phone \"@col[$_]\" duplicates in $dict/nonsilence_phones.txt (line $idx)\n"; $success = 0;}
        else {$nonsilence{@col[$_]} = 1;}
    }
    $idx ++;
}
close(NS);
$success == 0 || print "--> $dict/silence_phones.txt is OK\n";
print "\n";

# Checking disjoint -------------------------------
sub intersect {
    my ($a, $b) = @_;
    @itset = ();
    %itset = ();
    foreach(keys %$a) {
        if(exists $b->{$_} and !$itset{$_}) {
            push(@itset, $_);
            $itset{$_} = 1;
        }
    }
    return @itset;
}

print "Checking disjoint: silence_phones.txt, nonsilence_phones.txt\n";
@itset = intersect(\%silence, \%nonsilence);
if(@itset == 0) {print "--> disjoint property is OK.\n";}
else {$exit = 1; print "--> ERROR: silence_phones.txt and nonsilence_phones.txt has overlop: "; foreach(@itset) {print "$_ ";} print "\n";}
print "\n";

# Checking lexicon.txt -------------------------------
print "Checking $dict/lexicon.txt\n";
if(-z "$dict/lexicon.txt") {$exit = 1; print "--> ERROR: $dict/lexicon.txt is empty or not exists\n";}
if(!open(L, "<$dict/lexicon.txt")) {$exit = 1; print "--> ERROR: fail to open $dict/lexicon.txt\n";}
$idx = 1;
$success = 1;
print "--> reading $dict/lexicon.txt\n";
while(<L>) {
    chomp;
    my @col = split(" ", $_);
    $word = shift @col;
    foreach(0 .. @col-1) {
        if(!$silence{@col[$_]} and !$nonsilence{@col[$_]}) {
            $exit = 1; print "--> ERROR: phone \"@col[$_]\" is not in {, non}silence.txt (line $idx)\n"; 
            $success = 0;
        }
    }
    $idx ++;
}
close(L);
$success == 0 || print "--> $dict/lexicon.txt is OK\n";
print "\n";

# Checking extra_questions.txt -------------------------------
print "Checking $dict/extra_questions.txt ...\n";
if(-s "$dict/extra_questions.txt") {
    if(!open(EX, "<$dict/extra_questions.txt")) {$exit = 1; print "--> ERROR: fail to open $dict/extra_questions.txt\n";}
    $idx = 1;
    $success = 1;
    print "--> reading $dict/extra_questions.txt\n";
    while(<EX>) {
        chomp;
        my @col = split(" ", $_);
        foreach(0 .. @col-1) {
            if(!$silence{@col[$_]} and !$nonsilence{@col[$_]}) {
                $exit = 1; print "--> ERROR: phone \"@col[$_]\" is not in {, non}silence.txt (line $idx, block ", $_+1, ")\n"; 
                $success = 0;
            }
        }
        $idx ++;
    } 
    close(EX);
    $success == 0 || print "--> $dict/extra_questions.txt is OK\n";
} else {print "--> $dict/extra_phones.txt is empty\n";}

if($exit == 1) {exit 1;}
