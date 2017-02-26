#!/usr/bin/env perl
#===============================================================================
# Copyright 2016  (Author: Yenda Trmal <jtrmal@gmail.com>)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

my $Usage = <<EOU;
Filters the input by and semi-algebraic expression on the categories.

Usage: $0 [options] <categories-file> <expression>
 e.g.: cat data/dev10h.pem/kws/keywords.int | \
       $0 data/dev10h.pem/kws/categories "Characters>10&&NGramOrder=2"

Allowed options:
  -f <k>             : assume the KWID (for which the filter expression is
                         evaluated) on k-th column (int, default 0)

NOTE:
  When the expression is empty (or missing), its evaluated as always true,
  i.e. no entry will be removed from the input

CAVEATS:
  The operator '=' is equivalent to '=='.

  Do not use '-' character in the categories file if you want to use that
  category in the filter expression.  For example, the default setup adds
  the KWID itself as a category. In case you will use the Babel-style KWIDS,
  i.e. for example KW304-0008, you won't be able to use the KWID in
  the expression itself (but you can still filter according to other categories)
  i.e. for example
      KW306-0008&&OOV=1   might be a valid expression but most probably wont do
                          what you want (it will get parsed as
                          KW306 - (8 && (OOV == 1))  which is most probably not
                          what you wanted.
  Currently, there is no way how to make it work -- unless you rename
  the categories (i.e. for example substitute '-' by '_'. While this might be
  probably solved by taking the categories into account during parsing, it's
  probably not that important.

EOU

use strict;
use warnings 'FATAL';
use utf8;
use Switch;
use Data::Dumper;
use Scalar::Util qw(looks_like_number);
use Getopt::Long;
use POSIX;

my $debug = '';
my $field = 0;

GetOptions("debug" => \$debug,
           "f"     => \$field) || do {
  print STDERR "Cannot parse the command line parameters.\n\n";
  print $Usage . "\n";
  die "Cannot continue";
};

if ((@ARGV < 1) || (@ARGV>2)) {
  print STDERR "Incorrect number of parameters.\n\n";
  print $Usage . "\n";
  die "Cannot continue";
}

my $group_file = $ARGV[0];
my $str_expr="";
$str_expr=$ARGV[1] if defined($ARGV[1]);

# Split the expression into tokens (might need some more attention
# to make it really correct
sub tokenize_string {
  my $s = shift;
  $s =~ s/^\s+|\s+$//g;
  my @tokens = split(/ *(\&\&|\|\||\>\=|\<\=|==|!=|[\+\-\=\(\)\<\>\*\/^!]) */, $s);
  #print STDERR join(", ", @tokens) . "\n";
  return @tokens;
}



# precedence table should reflect the precedence of the operators in C
my %precedence = (
  #unary operators
  'u+' => 11,
  'u-' => 11,
  'u!' => 11,

  '^' => 10,
  #'(' => 10,
  #')' => 10,


  #arithmetic operators
  '*' => 8,
  '/' => 8,
  '%' => 8,

  '+' => 7,
  '-' => 7,

  # logical operators
  '<' => 5,
  '>' => 5,
  '>=' => 5,
  '<=' => 5,
  '=' => 4,
  '==' => 4,
  '!=' => 4,
  '&&' => 3,
  '||' => 2,
);

my %right=(
  #unary operators
  'u+' => 1,
  'u-' => 1,
  'u!' => 1,

  # this contradicts matlab, but it's what the mathematician's
  # interpretation is: 2^3^4 = 2^(3^4), instead of matlabs
  # left associativity 2^3^4 = (2^3)^4
  # as always -- if the order is important, use parentheses
  '^' => 1,
);

sub assoc {
  my $op = $_[0];
  return (exists $right{$op}) ? $right{$op} : -1;
}

sub looks_like_variable {
  return $_[0] =~ /^[A-Za-z_][A-Za-z_0-9]*$/;
}

sub unary_op {
  my $token = shift;
  my $op = shift;
  my $res;

  switch( $token ) {
    case 'u+' {$res =  $op}
    case 'u-' {$res = -$op}
    case 'u!' {$res = !$op}
    else {die "Unknown operator $token"}
  }

  return $res;
}

sub binary_op {
  my $token = shift;
  my $op2 = shift;
  my $op1 = shift;
  my $res;

  $op2 += 0.0;
  $op1 += 0.0;
  switch( $token ) {
    case '^'  {$res = $op1 ** $op2}
    case '*'  {$res = $op1 * $op2}
    case '/'  {$res = $op1 / $op2}
    case '%'  {$res = $op1 % $op2}
    case '+'  {$res = $op1 + $op2}
    case '-'  {$res = $op1 - $op2}
    case '<'  {$res = $op1 < $op2}
    case '>'  {$res = $op1 > $op2}
    case '>=' {$res = $op1 >= $op2}
    case '<=' {$res = $op1 <= $op2}
    case '='  {$res = $op1 == $op2}
    case '==' {$res = $op1 == $op2}
    case '!=' {$res = $op1 != $op2}
    case '&&' {$res = $op1 && $op2}
    case '||' {$res = $op1 || $op2}
    else {die "Unknown operator $token"}
  }

  return $res;
}

# refer to https://en.wikipedia.org/wiki/Shunting-yard_algorithm
# plus perl implementation in http://en.literateprograms.org/Shunting_yard_algorithm_(Perl)
sub to_postfix {
  my @stack;
  my @output = ();
  my $last = "";

  my @tokens=tokenize_string(shift);

  foreach my $token (@tokens) {
    next unless $token ne '';

    # detection of an unary operators
    # not sure if this heuristics is complete
    if (($token =~ /^[-+!]$/) &&
        (defined($precedence{$last}) || ($last eq '') || ($last eq ')')))  {
      #print "Unary op: $token\n";
      $token="u$token";
    }

    if (looks_like_number($token)) {
      if (looks_like_number($last) || looks_like_variable($last)) {
        die "Value tokens must be separated by an operator";
      }
      push @output, $token;
    } elsif (looks_like_variable($token)) {
      if (looks_like_number($last) || looks_like_variable($last)) {
        die "Value tokens must be separated by an operator";
      }
      push @output, $token;
    } elsif (defined $precedence{$token}) {
      my $p = $precedence{$token};

      while (@stack) {
        my $old_p = $precedence{$stack[-1]};
        last if $p > $old_p;
        last if $p == $old_p and (assoc($token) >= 0);
        push @output, pop @stack;
      }
      push @stack, $token;
    } elsif ($token eq '(') {
      push @stack, $token;
    } elsif ($token eq ')') {
      my $t;
      do {
        $t=pop @stack;
        push @output, $t unless $t eq '('
      } while ($t && ($t ne '('));
      die "No matching (" unless $t eq '(';
      #print "stack=[" . join(", ", @stack) . "] output=[" . join(", ", @output) . "]\n" ;
    } else {
      print "stack=[" . join(", ", @stack) . "] output=[" . join(", ", @output) . "]\n" ;
      die "Unknown token \"$token\" during parsing the expression";
    }
    $last=$token;
  }

  # dump the rest of the operators
  while (@stack) {
    my $t = pop @stack;
    die "No matching )" if $t eq '(';
    push @output, $t;
  }

  # final postfix expression
  return @output;
}

# this follows the standard RPM (postfix) expression evaluation
# the only possibly slightly confusing part is that when we encounter
# a variable, we lookup it's value in %vars. By default, (i.e. if the variable
# is not preset in the dict), the variable evaluates to 0 (false)
sub evaluate_postfix {
  my @expression = @{$_[0]};
  my %vars= %{$_[1]};

  my @stack = ();
  foreach my $token (@expression) {
    if (looks_like_number($token)) {
      push @stack, $token;
    } elsif (looks_like_variable($token)) {
      my $val = 0;
      if (defined $vars{$token}) {
        $val = $vars{$token};
      }
      push @stack, $val;
    } elsif (defined $precedence{$token}) {
      my $res;
      if ( $token =~ /^u.*$/) {
        my $op = pop @stack;
        $res = unary_op($token, $op);
      } else {
        my $op1 = pop @stack;
        my $op2 = pop @stack;
        $res = binary_op($token, $op1, $op2);
      }
      push @stack, $res;
    } else {
      die "Unknown token: $token, expression=[" . join(" ", @expression) . "]\n";
    }
    #print STDERR "token = $token; stack = [" . join(' ', @stack) . "]\n";

  }
  if (@stack != 1) {
    my $expr = join(" ", @expression);
    print STDERR "expression = [$expr]; stack = [" . join(' ', @stack) . "]\n";
    die "The operators did not reduce the stack completely!" if @stack != 1;
  }
  return pop @stack;
}


#--print "infix = [" . join(' ', @tokens) . "]\n";
#--my @exp = to_postfix(@tokens);
#--my %vals = (A=>50, C => -3);
#--print "output = [" . join(' ', @exp) . "]\n";
#--
#--print evaluate_postfix(\@exp, \%vals);


my @expression = to_postfix($str_expr);

my %GROUPS;
#Read the groups table
open(G, $ARGV[0]) or die "Cannot open the group table $ARGV[0]";
while (my $line = <G>) {
  my @entries = split(" ", $line);
  my $kwid = shift @entries;

  foreach my $group (@entries) {
    my @entries = split "=", $group;
    if (@entries == 2) {
      $GROUPS{$kwid}->{$entries[0]} = $entries[1];
    } elsif (@entries ==1 ) {
      $GROUPS{$kwid}->{$group} = 1;
    } else {
      die "Unknown format of the category $group";
    }
  }
}
close(G);

my $let_all_pass=0;
if (not @expression) {
  $let_all_pass=1;
}

while (my $line = <STDIN>) {
  #shortcut if the "ALL" groups is used
  if ($let_all_pass == 1) {
    print $line;
    next;
  }

  my @entries = split(" ", $line);
  my $kwid = $entries[$field];

  my $res = evaluate_postfix(\@expression, $GROUPS{$kwid});
  if ($res) {
    print $line;
  } else {
    print STDERR "Not keeping: $line" if $debug;
  }

}


