#!/usr/bin/env perl

# This program parses the output of nnet3-am-info or nnet3-info,
# and prints out a number between zero and one that reflects
# how saturated the (sigmoid and tanh) nonlinearities are, on average
# over the model.
#
# This is based on the 'avg-deriv' (average-derivative) values printed
# out for the sigmoid and tanh components.  The 'saturation' of such a component
# is defined as (1.0 - its avg-deriv / the maximum possible derivative of that nonlinearity),
# where the denominator is 1.0 for tanh and 0.25 for sigmoid.
# This component averages the saturation over all the sigmoid/tanh units in
# the network.
#
# It parses the Info() output of components of type SigmoidComponent,
# TanhComponent, and LstmNonlinearityComponent.  It prints an error message to
# stderr and returns with status 1 if it could not find the info for any such components
# in the input stream.

# Usage: nnet3-am-info 10.mdl | steps/nnet3/get_saturation.pl
# or: nnet3-info 10.raw | steps/nnet3/get_saturation.pl

use warnings;

my $num_nonlinearities = 0;
my $total_saturation = 0.0;

while (<STDIN>) {
  if (m/type=SigmoidComponent/) {
    # a line like:
    # component name=Lstm1_f type=SigmoidComponent, dim=1280, count=5.02e+05,
    # value-avg=[percentiles(0,1,2,5 10,20,50,80,90
    # 95,98,99,100)=(0.06,0.17,0.19,0.24 0.28,0.33,0.44,0.62,0.79
    # 0.96,0.99,1.0,1.0), mean=0.482, stddev=0.198],
    # deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90
    # 95,98,99,100)=(0.0001,0.003,0.004,0.03 0.12,0.18,0.22,0.24,0.25
    # 0.25,0.25,0.25,0.25), mean=0.198, stddev=0.0591]
    if (m/deriv-avg=.+mean=([^,]+),/) {
      $num_nonlinearities += 1;
      my $this_saturation = 1.0 - ($1 / 0.25);
      $total_saturation += $this_saturation;
    } else {
      print STDERR "$0: could not make sense of line (no deriv-avg?): $_";
    }
  } elsif (m/type=TanhComponent/) {
    if (m/deriv-avg=.+mean=([^,]+),/) {
      $num_nonlinearities += 1;
      my $this_saturation = 1.0 - ($1 / 1.0);
      $total_saturation += $this_saturation;
    } else {
      print STDERR "$0: could not make sense of line (no deriv-avg?): $_";
    }
  } elsif (m/type=LstmNonlinearityComponent/) {
    # An example of a line like this is right at the bottom of this program, it's extremely long.
    my $ok = 1;
    foreach my $sigmoid_name ( ("i_t", "f_t", "o_t") ) {
      if (m/${sigmoid_name}_sigmoid={[^}]+deriv-avg=[^}]+mean=([^,]+),/) {
        $num_nonlinearities += 1;
        my $this_saturation = 1.0 - ($1 / 0.25);
        $total_saturation += $this_saturation;
      } else {
        $ok = 0;
      }
    }
    foreach my $tanh_name ( ("c_t", "m_t") ) {
      if (m/${tanh_name}_tanh={[^}]+deriv-avg=[^}]+mean=([^,]+),/) {
        $num_nonlinearities += 1;
        my $this_saturation = 1.0 - ($1 / 1.0);
        $total_saturation += $this_saturation;
      } else {
        $ok = 0;
      }
    }
    if (! $ok) {
      print STDERR "Could not parse at least one of the avg-deriv values in the following info line: $_";
    }
  }
}


if ($num_nonlinearities == 0) {
  print "0.0\n";
  exit(1);
} else {
  my $saturation = $total_saturation / $num_nonlinearities;
  if ($saturation < 0.0 || $saturation > 1.0) {
    print STDERR "Bad saturation value: $saturation\n";
    exit(1);
  } else {
    print "$saturation\n";
  }
}




# example line with LstmNonlinearityComponent that we parse:
# component name=lstm2.lstm_nonlin type=LstmNonlinearityComponent, input-dim=2560, output-dim=1024, learning-rate=0.002, max-change=0.75, cell-dim=512, w_ic-rms=0.9941, w_fc-rms=0.8901, w_oc-rms=0.9794, count=3.53e+05, i_t_sigmoid={ self-repair-lower-threshold=0.05, self-repair-scale=1e-05, self-repaired-proportion=0.0722299, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.04,0.08,0.09,0.12 0.17,0.25,0.46,0.76,0.87 0.91,0.96,0.96,1.0), mean=0.494, stddev=0.253], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.0007,0.03,0.04,0.06 0.09,0.12,0.19,0.23,0.24 0.25,0.25,0.25,0.25), mean=0.179, stddev=0.0595] }, f_t_sigmoid={ self-repair-lower-threshold=0.05, self-repair-scale=1e-05, self-repaired-proportion=0.0688061, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.06,0.11,0.13,0.17 0.22,0.30,0.51,0.70,0.82 0.90,0.96,0.98,1.0), mean=0.509, stddev=0.219], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.001,0.01,0.03,0.07 0.11,0.15,0.21,0.24,0.25 0.25,0.25,0.25,0.25), mean=0.194, stddev=0.0561] }, c_t_tanh={ self-repair-lower-threshold=0.2, self-repair-scale=1e-05, self-repaired-proportion=0.178459, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(-1.0,-0.98,-0.97,-0.92 -0.82,-0.65,-0.01,0.66,0.87 0.94,0.95,0.97,0.99), mean=0.00447, stddev=0.612], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.003,0.02,0.04,0.10 0.14,0.25,0.65,0.84,0.90 0.94,0.97,0.97,0.98), mean=0.58, stddev=0.281] }, o_t_sigmoid={ self-repair-lower-threshold=0.05, self-repair-scale=1e-05, self-repaired-proportion=0.0608838, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.02,0.07,0.09,0.12 0.17,0.25,0.52,0.77,0.86 0.90,0.94,0.96,0.99), mean=0.514, stddev=0.256], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.007,0.04,0.04,0.07 0.09,0.12,0.19,0.23,0.24 0.25,0.25,0.25,0.25), mean=0.175, stddev=0.0579] }, m_t_tanh={ self-repair-lower-threshold=0.2, self-repair-scale=1e-05, self-repaired-proportion=0.134653, value-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(-0.99,-0.95,-0.92,-0.85 -0.73,-0.51,0.02,0.48,0.73 0.86,0.96,0.98,1.0), mean=0.00581, stddev=0.522], deriv-avg=[percentiles(0,1,2,5 10,20,50,80,90 95,98,99,100)=(0.002,0.03,0.04,0.13 0.26,0.41,0.75,0.93,0.97 0.99,1.0,1.0,1.0), mean=0.672, stddev=0.272] }
