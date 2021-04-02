#!/usr/bin/perl

# Note from Pawel: This file has been modified from its original version to account for
# extracting mappings for distant channels, as there are some exceptions in mapping between
# micropones types and physical channels across some meetings.

# An example ICSI file looks like this
# <Meeting Session="Bed002" DateTimeStamp="2001-02-05-1500" Version="2" VersionDate="Jan 7 2004">
# 
#   <Preamble>
#     <Notes>
#       me011 left early (11 minutes into the meeting).
#    fe004 arrived late (15.5 minutes into the meeting),
#    and did not read digits.
#    
#    Channel 3 mic was on, lying on the table, but unused.
#    
#    Channel 8 had a huge amount of static noise toward the
#    latter part of the meeting.  (Andrea mic)
#    
#    Laptop presentation.
#    
#    Some Discourse highlights:
#    Nice me010 / mn015 interaction at minute 22-22:26
#    (misunderstanding and clarification)
#    Nice me010 / mn015 overlapping/duetting
#    minute 26:26-27:14
#     </Notes>
#     <Channels>
#       <Channel Name="chan0" Mic="l1" Transmission="w1" Gain="8" AudioFile="chan0.sph"/>
#       ...
#     </Channels>
#     <Participants DB="icsi1.spk">
#       <Participant Name="me011" Channel="chan0"/>
#       <Participant Name="mn015" Channel="chan1"/>
#       <Participant Name="me010" />
#       ...
#     </Participants>
#   </Preamble>
# 
# 
#   <Transcript StartTime="0.0" EndTime="3915.955">
# 
#     <Segment StartTime="5.750" EndTime="6.660" Participant="me003">
#       <Uncertain> I guess. </Uncertain>
#     </Segment>
#    <Segment StartTime="6.323" EndTime="7.124" Participant="me011">
#       O_K, we're on.
#    </Segment>
#     </Segment>
#     <Segment StartTime="10.303" EndTime="12.448" Participant="me011">
#       So just make sure that th- your wireless mike is on,
#     </Segment>
#     <Segment StartTime="12.448" EndTime="13.895" Participant="me011">
#       if you're wearing a wireless.
#     </Segment>
#     <Segment StartTime="12.602" EndTime="14.920" Participant="me003">
#       Check one. Check one.
#     </Segment>
#     <Segment StartTime="14.655" EndTime="16.180" Participant="me011">
#       <NonVocalSound Description="door slams"/>
#     </Segment>
#     <Segment StartTime="14.734" EndTime="15.900" Participant="me010">
#       <VocalSound Description="outbreath"/>
#     </Segment>
#     <Segment StartTime="14.905" EndTime="15.716" CloseMic="false">
#       <NonVocalSound Description="door slams"/>
#     </Segment>
#     <Segment StartTime="16.061" EndTime="17.450" Participant="me003">
#       <VocalSound Description="laugh"/>
#     </Segment>
#     <Segment StartTime="18.219" EndTime="23.139" Participant="me011">
#       And you should be able to see which one - which one you're on by, uh, watching the little bars change.
#     </Segment>
#     <Segment StartTime="23.825" EndTime="25.291" Participant="mn015">
#       So, which is <Emphasis> my </Emphasis> bar? <Pause/> Mah! <Comment Description="interjection"/>
#     </Segment>
#     <Segment StartTime="25.291" EndTime="28.575" Participant="mn015">
#       Number one. <VocalSound Description="laugh"/>
#     </Segment>
#     <Segment StartTime="26.237" EndTime="26.693" Participant="me011">
#       Yep.
#     </Segment>
#     <Segment StartTime="26.310" EndTime="27.481" Participant="me003">
#       <VocalSound Description="laugh"/>
#     </Segment>
#     ...

use XML::LibXML;
use Data::Dumper;

#use strict;
use warnings;

if (@ARGV != 2) {
  print STDERR "Usage: icsi_parse_transcripts.pl <patth-to-meet-xml> <out-dir>\n";
  exit(1);
}

my $meet_file=shift @ARGV;
my $out_file=shift @ARGV;

open(S, "<$meet_file") || die "opening meeting file $meet_file";
my $parser = XML::LibXML->new();
my $xmldoc = $parser->parse_file($meet_file);
close(S);

$meetid='';
for my $m ($xmldoc->findnodes('/Meeting')) {
  $meetid = $m->getAttribute('Session');
  last;
}

#build spk2chan map
my %spk2chan = ();
for my $p ($xmldoc->findnodes('/Meeting/Preamble/Participants/Participant')) {
   my $spk = $p->getAttribute("Name");
   my $chan = 'chanX'; #some speakers do not have headsets, back-off to distant channel in that case
   if ($p->hasAttribute('Channel')) {
     $chan = $p->getAttribute("Channel");
   } else {
     print "Warning: $spk does not have headset. Backing off to distant mics [chanX].\n"
   }
   $spk2chan{$spk} = $chan; 
}

print 'Info: spk2chan hash : ', Dumper(\%spk2chan), "\n";

#build mic2chan map
my %mic2chan = ();
for my $p ($xmldoc->findnodes('/Meeting/Preamble/Channels/Channel')) {
   my $mic = $p->getAttribute("Mic");
   my $chan = $p->getAttribute("Name");
   if (!defined $mic2chan{$mic}) {
     $mic2chan{$mic} = $chan;
   } else {
     $mic2chan{$mic} = "$mic2chan{$mic},$chan";
   }
}


#parse segments and produce and store them in the list
#MeetingId chan spk start end transcript

my @segments = ();
for my $seg ($xmldoc->findnodes('/Meeting/Transcript/Segment')) {
  
   if (!$seg->hasAttribute('Participant')) {
     next;
   }

   my $spk = $seg->getAttribute('Participant');
   my $stime = $seg->getAttribute('StartTime') || die "Error: Missing StartTime argument in $seg";
   my $etime = $seg->getAttribute('EndTime') || die "Error: Missing EndTime argument in $seg";
   my $text = $seg->textContent();

   $text =~ s/^\s+//;
   $text =~ s/\s+$//;

   #print "\n", $spk,"  ",$stime," ",$etime," ",$text,"\n";

   my %segment = ();
   $segment{'spk'}=$spk;
   $segment{'stime'}=$stime;
   $segment{'etime'}=$etime;
   $segment{'text'}=$text;
   push (@segments, \%segment);
}

open(W, ">$out_file") || die "opening output file $out_file";
for my $i (0 .. $#segments) {
  
  my $spk = $segments[$i]->{'spk'};
  my $stime = $segments[$i]->{'stime'};
  my $etime = $segments[$i]->{'etime'};
  my $text = $segments[$i]->{'text'};
  my $chan = $spk2chan{$spk};
  my $chanpzm = $mic2chan{"c2"}; #for ICSI recipe, we only use D* PZM mics (c2)
  
  print W "$meetid $chan $chanpzm $spk $stime $etime $text \n";
}
close(W);

