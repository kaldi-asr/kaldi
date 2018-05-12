#!/usr/bin/perl


#the input format should be:
#meetid chan spk stime etime transcripts....
#here we renormalize transcripts 

$field_begin=6;

while (<>) {  

  chomp ($_);
  @A = split(" ", $_);
  if ($#A < $field_begin) { next; } #empty transcript

  $text = join(" ", @A[$field_begin..$#A]);
  #make uppercase
  $text = uc $text;
  #remove censored captions
  $text =~ s/[\@]//g;
  #remove puncation signs !?.,
  $text =~ s/[\!\?\.\,"\:]//g;
  # O_K to OKAY, C_C'ED to C. C. 'ED
  $text =~ s/O\_K/OKAY/g;
  $text =~ s/C\_C\'ED/C. C. 'ED/g;
  #change spelled words from X_M_L to X_M_L_, note the last sign, i.e. L may be the last one in line
  $text = "$text\n";
  $text =~ s/([A-Z][A-Z\_]*\_[A-Z\-])[\s\n]+/$1_ /g;
  #and then renormalize to X. M. L.
  $text =~ s/\_/. /g;
  #there is couple of strange 1x, 2x...4x entries, I tried to listen to these, but cannot figure out what they mean - nullify them
  $text =~ s/\s*[1-4]X[\s\n]+//g;
  # remove beginning '-' from i.e. -WORD-C.
  $text =~ s/\s*\-([A-Z\'\-\.]+)[\s\n]+/ $1 /g;
  # remove standalone -
  $text =~ s/\s+\-\s+/ /g;
  $text =~ s/^\-\s+//g;
  $text =~ s/ - / /g;
  $text =~ s/\s+\-$//g;
  #normalise/remove white spaces
  $text =~ s/^\s*//g;
  $text =~ s/\s*$//g;
  $text =~ s/\s+/ /g;

  if (length ($text) < 1) { next; }

  $header = join(" ", @A[0..$field_begin-1]);
  print "$header $text\n";
 
}



