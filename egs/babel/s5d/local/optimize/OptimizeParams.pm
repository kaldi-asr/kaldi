# Author: Jason Eisner, Univ. of Pennsylvania
#
# $Revision: 3.11 $ of $Date: 2006/04/12 08:53:23 $

# !!! should add root-finding methods with derivative (newton-raphson:
# use rtsafe, section 9.4) and in multiple dimensions (sections 9.5, 9.6).

package OptimizeParams;
use strict;

BEGIN {
  use Exporter ();
  use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
  $VERSION = do { my @r = (q$Revision: 3.11 $ =~ /\d+/g); sprintf "%d."."%02d" x $#r, @r }; # must be all one line, for MakeMaker

  @ISA       = qw(Exporter);
  @EXPORT_OK = qw(&powell &easybrent &easydbrent &easyzbrent
  &mnbrak &brent &dbrent &zbrent
  $machine_epsilon $inf &basisvectors);
}

# A sample program with simple examples on a one-dimensional function.
#
# #!/usr/local/bin/perl5 -w
#
# use OptimizeParams qw(&powell &easybrent &easydbrent &zbrent);
# use strict 'vars';
#
# sub f { sin(($_[0]-12.34567)/8)**2-0.5 }   # function
# sub df { sin(2*($_[0]-12.34567)/8)/8 }     # derivative
# sub fdf { my($temp)=($_[0]-12.34567)/8;    # (function, derivative) computed at one go
# 	      (sin($temp)**2-0.5, sin(2*$temp)/8) }
#
# # Three ways to find (x,f(x)) at minimum of function, namely (12.34567,-0.5)
# print join(" ",easybrent(0,1,\&f)), "\n";
# print join(" ",easydbrent(0,1,\&f,\&df)), "\n";
# print join(" ",easydbrent(0,1,\&fdf)), "\n";
#
# # A fourth way, using a multidimensional optimizer even though f happens
# # to be 1-dimensional.  The vector [0] is our starting guess.
# my($xvec,$fx) = powell(\&f,[0]);
# print join(" ",@$xvec,$fx), "\n";
#
# # Find zero of function, namely 6.06
# my($x)=zbrent(\&f,0,13);  print $x," ",&f($x),"\n";

# ----------------------------------------------------------------------

use vars @EXPORT_OK;
$inf=exp(1e307);   # could just use the bareword inf, which seems to work but generates warnings with -w
$machine_epsilon = 1; $machine_epsilon /= 2 while 1 + $machine_epsilon/2 > 1;

sub FMAX {    # (maximum)
  $_[0] > $_[1] ? $_[0] : $_[1];
}

sub SIGN {
  $_[1] >= 0 ? abs($_[0]) : -abs($_[0]);
}


# Direction Set (Powell's) Methods in Multidimensions
# From Numerical Recipes in C, Section 10.5, p. 417ff.  Ported to Perl.
#
# Minimization of a function of n variables [for which the gradient is
# not known].  Required arguments are (a reference to) the function
# and (a reference to) a length-n vector holding the coordinates of
# the starting point. Optional arguments are a fractional tolerance in
# the output value (used as a stopping criterion), a fractional
# tolerance in the input value (used as a stopping criterion on
# one-dimensional searches), and (a reference to) a list of n
# (references to) such vectors, holding an initial set of directions.
# Return values are a reference to a vector holding the coordinates at
# the minimum; the value of the function at that minimum; the number
# of iterations taken; and the final set of directions.
#
# This Perl version has a few different representational conventions.
# It's now the ROWS of $xi (not the columns) that hold the direction vectors.
# And the coordinates are 0-indexed, not 1-indexed.
# The $itol argument is new.

sub powell {
  my($funcref,$p,$ftol,$iftol,$xi) = @_;
  my($n) = scalar @$p;   # Number of dimensions.
  my($ITMAX)=200;        # Maximum allowed iterations.

  # Defaults for optional arguments
  $ftol = $machine_epsilon unless defined $ftol;
  $iftol = 2.0e-4 unless defined $iftol;           # in the C version, this is TOL (defined at linmin)
  $xi = &basisvectors($n) unless (defined $xi);

  my($fret) = &$funcref(@$p);
  my(@pt) = @$p;                  # Save the initial point.
  my($iter);
  for($iter=1;;++$iter) {
    my($fp) = $fret;
    my($ibig) = 0;
    my($del) = 0;                 # Will be the biggest function decrease.
    my($i);
    for ($i=0;$i<$n;$i++) {      # In each iteration, loop over all directions in the set.
      my($xit) = \@{$xi->[$i]};        # Copy the direction,
      my($fptt) = $fret;
      $fret = &linmin($p,$xit,$funcref,$iftol);  # minimize along it,
      if (abs($fptt-$fret) > $del) {             # and record it if it is the largest decrease so far.
        $del=abs($fptt-$fret);
        $ibig=$i;
      }
    }
    if (2*abs($fp-$fret) <= $ftol*(abs($fp)+abs($fret))) {  # Termination criterion.
      return($p,$fret,$iter,$xi);
    }
    die "$0: powell exceeding maximum of $ITMAX iterations" if ($iter==$ITMAX);

    {
      my($xit);
      my(@ptt);
      my($j);
      for ($j=0;$j<$n;$j++) {    # Construct the extrapolated point and the average direction moved.  Save the old starting point.
        $ptt[$j] = 2*$p->[$j] - $pt[$j];
        $xit->[$j] = $p->[$j] - $pt[$j];
        $pt[$j] = $p->[$j];
      }
      my($fptt) = &$funcref(@ptt);
      if ($fptt < $fp) {
        my($t) = 2 * ($fp-2*$fret+$fptt) * ($fp-$fret-$del)**2 - $del*($fp-$fptt)**2;
        if ($t < 0) {
          $fret = &linmin($p,$xit,$funcref);
          $xi->[$ibig] = $xi->[$n-1];
          $xi->[$n-1] = $xit;
        }
      }
    }
  }         # Back for another iteration

  die "$0: internal error in powell: should never have reached this line";
}

sub basisvectors {      # returns the basis vectors in the given dimension (a reference to a list of references to lists)
  my($n) = @_;
  my($vects);
  my($i,$j);
  for ($i=0;$i<$n;$i++) {
    for ($j=0;$j<$n;$j++) {
      $vects->[$i][$j] = ($i==$j ? 1 : 0);
    }
  }
  return $vects;
}



{
  my($ncom);          # "Global" variables for linmin to communicate with f1dim.
  my(@pcom, @xicom, $nrfuncref);

  # Routine called by powell.
  # From Numerical Recipes in C, Section 10.5, p. 419.  Ported to Perl.
  #
  # Given an n-dimensional point $p and an n-dimensional direction
  # vector $xi (both references to lists), moves and resets $p to
  # where the function $funcref takes on a minimum along the direction
  # $xi from $p, and replaces $xi by the actual vector displacement that
  # $p was moved.  Returns the value of $funcref at $p.  This is actually
  # all accomplished by calling the routines mnbrak and brent.
  # $iftol is a tolerance on the input value, passed to brent.

  sub linmin {
    my($p,$xi,$funcref,$iftol) = @_;

    print STDERR "$0: linmin: searching from (",join(", ",@$p),") in direction (",join(", ",@$xi),")\n";

    $ncom = @$p;              # Define the global variables.
    $nrfuncref = $funcref;
    @pcom = @$p;
    @xicom = @$xi;

    my($ax) = 0;              # Initial guess for brackets.
    my($xx) = 1;
    my($bx);
    ($ax,$xx,$bx) = &mnbrak($ax,$xx,\&f1dim);
    my($xmin,$fret) = &brent($ax,$xx,$bx,\&f1dim,$iftol);
    my($j);
    for ($j=0;$j<$ncom;$j++) {
      $p->[$j] += ($xi->[$j] *= $xmin);
    }
    return $fret;
  }

  # Function minimized by linmin.

  sub f1dim {
    my($x) = @_;
    my(@xt);
    my($j);
    for($j=0; $j<$ncom;$j++) {
      $xt[$j] = $pcom[$j] + $x * $xicom[$j];
    }
    return &$nrfuncref(@xt);
  }
}



# Easy way to call mnbrak and brent together in order to minimize
# a function.
#
# ax and bx are any distinct points; we'll look for a minimum in the
# downhill direction on the line through (ax,f(ax)) and (bx,f(bx)).
#
# Return value is the same as brent, namely (x,f(x)).  But we might
# fail to find a minimum!  If the function never increases again so
# far as we can tell -- it plateaus, or decreases toward infinity, or
# increases in a range that mnbrak doesn't sample -- then we'll return
# (+/-inf, minimum value we found).  Here the +/- is according to
# which direction we searched in, and the minimum value is f(x) for
# the last finite x we considered; this value may or may not be
# finite, but should indicate the asymptotic behavior of the function.
#
# Just as in brent, the tolerance $tol can be omitted.

sub easybrent {
  my($ax,$bx,$funcref,$tol) = @_;
  my($newa,$newb,$newc,$fa,$fb,$fc) = &mnbrak($ax,$bx,$funcref);
  return ($newc,$fb) if ($newc==$inf || $newc==-$inf);
  &brent($newa,$newb,$newc,$funcref,$tol);
}

# Easy way to call mnbrak and dbrent together in order to minimize
# a function whose derivative is known.
# ax and bx are any distinct points; we'll look for a minimum in the
# downhill direction on the line through (ax,f(ax)) and (bx,f(bx)).
#
# See easybrent for return value convention when we fail.
#
# Just as in dbrent, the tolerance $tol can be omitted.  So can
# $dfuncref, if $funcref returns a pair of values -- both the function
# and its derivative.

sub easydbrent {
  my($ax,$bx,$funcref,$dfuncref,$tol) = @_;
  my($newa,$newb,$newc,$fa,$fb,$fc) = &mnbrak($ax,$bx,$funcref);
  return ($newc,$fb) if ($newc==$inf || $newc==-$inf);
  &dbrent($newa,$newb,$newc,$funcref,$dfuncref,$tol);
  # If we want to check output against brent:
  # my(@ans1)=&dbrent($newa,$newb,$newc,$funcref,$dfuncref);
  # my(@ans2)=&brent($newa,$newb,$newc,$funcref);
  # die "dbrent $ans1[0], brent $ans2[0]\n" unless &main::near($ans1[0]+1e6,$ans2[0]+1e6);
  # @ans1;
}

# Easy way to TRY to bracket a root and then call zbrent to find the
# root.  The calling convention is similar to easybrent: we are given
# two starting points.  If they have different signs, we just call
# zbrent.  If they have the same sign and are both positive, we search
# in the downhill direction for a negative value (using mnbrak
# together with a modified golden-section minimizer (section 10.1)
# that stops as soon as it crosses zero).  Similarly, if they have the
# same sign and are both positive, we search uphill for a positive
# value.

sub easyzbrent {
  my($ax,$bx,$funcref) = @_;
  die "Not implemented yet; must call zbrent directly"
}


# Parabolic Interpolation and Brent's Method in one dimension
# From Numerical Recipes in C, Section 10.2, p. 404.  Ported to Perl.
#
# Given a continuous function of one variable referenced by $funcref,
# and given a bracketing triplet of abcissas $ax, $bx, $cx as returned
# by mnbrak, this routine isolates the minimum to a fractional
# precision of about $tol using Brent's method.  Returns (x, f(x)) at
# the minimum.  $tol is set to a good default if omitted.
#
# See easybrent for an easier way to call this.

sub brent {
  my($ax, $bx, $cx, $funcref, $tol) = @_;
  $tol = sqrt($machine_epsilon) unless defined $tol;
  my($e) = 0.0;                 # This will be the distance moved on the step before last.
  my($ITMAX) = 100;             # The maximum allowed number of iterations.
  my($CGOLD) = 0.3819660;       # The golden ratio.  [Actually, 1-golden ratio.]
  my($ZEPS) = 1.0e-10;

  my($a) =($ax < $cx ? $ax : $cx);   # a and b must be in ascending order, but input abscissas need not be.
  my($b) =($ax > $cx ? $ax : $cx);
  my($x,$w,$v);  $x=$w=$v=$bx;       # Initializations ...
  die "brent: inputs out of order\n" unless $a < $x && $x < $b;   # probably should also check f(x) < f(a),f(b)
  my($fw,$fv,$fx); ($fw)=($fv)=($fx)=&$funcref($x);
  my($d,$u,$fu);

  my($iter);
  for ($iter=1; $iter<=$ITMAX; $iter++) {      # Main program loop.
    my($xm) = 0.5*($a+$b);
    my($tol1)=$tol*abs($x)+$ZEPS;
    my($tol2)=2.0*$tol1;
    return ($x,$fx) if (abs($x-$xm) <= ($tol2-0.5*($b-$a)));   # Test for done here.
    if (abs($e) > $tol1) {                     # Construct a trial parabolic fit.
      my($r) = ($x-$w)*($fx-$fv);
      my($q) = ($x-$v)*($fx-$fw);
      my($p) = ($x-$v)*$q - ($x-$w)*$r;
      $q=2.0*($q-$r);
      $p = -$p if $q > 0;
      $q = abs($q);
      my($etemp)=$e;
      $e=$d;
      if (abs($p) >= abs(0.5*$q*$etemp) || $p <= $q*($a-$x) || $p >= $q*($b-$x)) {
        $d = $CGOLD*($e = ($x >= $xm ? $a-$x : $b-$x));
      }
      # The above conditions determine the acceptability of the parabolic
      # fit.  Here we take the golden section step into the larger of the two
      # segments.
      else {
        $d=$p/$q;      # Take the parabolic step.
        $u=$x+$d;
        $d = &SIGN($tol1,$xm-$x) if ($u-$a < $tol2 || $b-$u < $tol2);
      }
    } else {
      $d=$CGOLD*($e=($x >= $xm ? $a-$x : $b-$x));
    }
    $u = (abs($d) >= $tol1 ? $x+$d : $x+&SIGN($tol1,$d));
    ($fu) = &$funcref($u);    # This is the one function evaluation per iteration.
    if ($fu <= $fx) {       # Now decide what to do with our function evaluation.
      ($u >= $x ? $a : $b) = $x;
      ($v, $w, $x) = ($w, $x, $u);   # Housekeeping follows:
      ($fv, $fw, $fx) = ($fw, $fx, $fu);
    } else {
      ($u < $x ? $a : $b) = $u;
      if ($fu <= $fw || $w == $x) {
        $v=$w;
        $w=$u;
        $fv=$fw;
        $fw=$fu;
      } elsif ($fu <= $fv || $v == $x || $v == $w) {
        $v = $u;
        $fv = $fu;
      }
    }                                # Done with housekeeping.  Back for another iteration.
  }
  die "$0: brent: Maximum number of iterations ($ITMAX) exceeded";
}

# One-Dimensional Search with First Derivatives
# From Numerical Recipes in C, Section 10.3, p. 405.  Ported to Perl.
#
# Given a continuous function of one variable referenced by $funcref,
# and its derivative referenced by $dfuncref, and given a bracketing
# triplet of abcissas $ax, $bx, $cx as returned by mnbrak, this
# routine isolates the minimum to a fractional precision of about $tol
# using a modification of Brent's method that uses derivatives.
# Returns (x, f(x)) at the minimum.  $tol is set to a good default if
# omitted.
#
# See easydbrent for an easier way to call this.

sub dbrent {
  my($ax, $bx, $cx, $funcref, $dfuncref, $tol) = @_;
  $tol = sqrt($machine_epsilon) unless defined $tol;

  my($e) = 0.0;                 # This will be the distance moved on the step before last.
  my($ITMAX) = 100;             # The maximum allowed number of iterations.
  my($ZEPS) = 1.0e-10;

  my($a) =($ax < $cx ? $ax : $cx);   # a and b must be in ascending order, but input abscissas need not be.
  my($b) =($ax > $cx ? $ax : $cx);
  my($w,$v,$x,$u);  $w=$v=$x=$bx;    # Initializations ...
  die "dbrent: inputs out of order\n" unless $a < $x && $x < $b;    # probably should also check f(x) < f(a),f(b)
  my($fx,$dx)=&$funcref($x);
  $dx=&$dfuncref($x) unless defined $dx;   # if $funcref only returned one value in previous line
  my($fw,$fv,$fu); $fw=$fv=$fx;
  my($dw,$dv,$du); $dw=$dv=$dx;      # All our housekeeping chores are doubled by the necessity of moving derivative values around as well as function values.
  my($d);

  my($iter);
  for ($iter=1; $iter<=$ITMAX; $iter++) {      # Main program loop.
    my($xm) = 0.5*($a+$b);
    my($tol1)=$tol*abs($x)+$ZEPS;
    my($tol2)=2.0*$tol1;
    # print "a $a b $b x $x xm $xm\n";
    return ($x,$fx) if (abs($x-$xm) <= ($tol2-0.5*($b-$a)));   # Test for done here.
    if (abs($e) > $tol1) {                       # Construct a trial parabolic fit.
      my($d1)=2.0*($b-$a);                         # Initialize these d's to an out-of-bracket value
      my($d2)=$d1;
      $d1 = ($w-$x)*$dx/($dx-$dw) if ($dw != $dx);  # Secant method with one point.
      $d2 = ($v-$x)*$dx/($dx-$dv) if ($dv != $dx);  # And the other.
      # Which of these two estimates of d shall we take?
      # We will insist that they be within the bracket, and on
      # the side pointed to by the derivative at x:
      my($u1)=$x+$d1;
      my($u2)=$x+$d2;
      my($ok1) = ($a-$u1)*($u1-$b) > 0 && $dx*$d1 <= 0;
      my($ok2) = ($a-$u2)*($u2-$b) > 0 && $dx*$d2 <= 0;
      my($olde) = $e;                          # Movement on the step before last.
      $e = $d;
      if ($ok1 || $ok2) {                      # Take only an acceptable d, and if both are acceptable, then take the smallest one.
        if ($ok1 && $ok2) {
          $d=(abs($d1) < abs($d2) ? $d1 : $d2);
        } elsif ($ok1) {
          $d=$d1;
        } else {
          $d=$d2;
        }
        if (abs($d) <= abs(0.5*$olde)) {
          $u=$x+$d;
          $d=&SIGN($tol1,$xm-$x) if ($u-$a < $tol2 || $b-$u < $tol2);
        } else {                               # Bisect, not golden section.
          $d=0.5*($e=($dx >= 0 ? $a-$x : $b-$x));  # Decide which segment by the sign of the derivative.
        }
      } else {
        $d=0.5*($e=($dx >= 0 ? $a-$x : $b-$x));
      }
    } else {
      $d=0.5*($e=($dx >= 0 ? $a-$x : $b-$x));
    }
    if (abs($d) >= $tol1) {
      $u=$x+$d;
      ($fu,$du)=&$funcref($u);
    } else {
      $u=$x+&SIGN($tol1,$d);
      ($fu,$du)=&$funcref($u);
      return ($x,$fx) if ($fu > $fx);  # If the minimum step in the downhill direction takes us uphill, then we are done.
    }
    # Now all the housekeeping, sigh.
    $du=&$dfuncref($u) unless defined $du;   # if $funcref only returned one value just above
    if ($fu <= $fx) {
      ($u >= $x ? $a : $b) = $x;
      ($v,$fv,$dv)=($w,$fw,$dw);
      ($w,$fw,$dw)=($x,$fx,$dx);
      ($x,$fx,$dx)=($u,$fu,$du);
    } else {
      ($u < $x ? $a : $b) = $u;
      if ($fu <= $fw || $w==$x) {
        ($v,$fv,$dv)=($w,$fw,$dw);
        ($w,$fw,$dw)=($u,$fu,$du);
      } elsif ($fu < $fv || $v == $x || $v == $w) {
        ($v,$fv,$dv)=($u,$fu,$du);
      }
    }
  }
  die "$0: dbrent: Maximum number of iterations ($ITMAX) exceeded\n";
  # Alternative:
  # warn "$0: dbrent: Maximum number of iterations ($ITMAX) exceeded.  Trying brent ...\n";
  # &brent($ax,$bx,$cx,$funcref,$tol);
}


# Routine for Initially Bracketing a Minimum.
# From Numerical Recipes in C, Section 10.1, p. 400.  Ported to Perl.
#
# Given a continuous function referenced by $funcref, and distinct
# initial points $ax and $bx, this routine searches in the downhill
# direction (defined by the function as evaluated at the initial
# points) and returns new points $ax, $bx, $cx that bracket a minimum
# of the function [in the sense that b is between a and c, and f(b) is
# less than both f(a) and f(c)]. Also returned are the function values
# at the three points, $fa, $fb, and $fc.
#
# JME: If $cx is +inf (resp. -inf), this means that we searched in the
# positive (resp. negative) direction and the function just decreased
# forever (either to a plateau or without bound - look at $fb to see
# the last finite value).  At least, it decreased at all the points
# where we sampled it - we might have skipped right over a spike.  So
# either there is no minimum in the direction we searched, or we
# missed it; in either case our return values won't bracket any minimum
# and the caller should either give up or try something else!
#
# JME: Note that it's also possible that $cx remains finite, but that
# the minimum $fb that we bracket is -$inf (and typically $fc will be
# -$inf too).
#
# JME: f(b) is now required to be STRICTLY less than f(a) and f(c).
# This avoids counting an "extended" point of inflection as a minimum.
# I imagine the minimization routines would nonetheless be willing to
# find such if it's in the interval (should check...), but requiring
# us to search past it here is important for the previous paragraph:
# if the function value is eventually -inf forever due to overflow, we
# still keep searching forever until the abcissa is also +/- inf,
# rather than saying we've hit a plateau and that's enough to stop.
#
# It's ok if &$funcref returns multiple values; we'll evaluate it in
# list context and use only the first value.  This is useful because
# of the calling convention for dbrent; e.g., easydbrent relies on it.

sub mnbrak {
  my($ax, $bx, $funcref) = @_;
  my($GOLD) = 1.618034;
  my($GLIMIT) = 100.0;
  my($TINY) = 1.0e-20;

  die "mnbrak: $ax and $bx must be different\n" if $ax==$bx;  # JME: added
  my($fa) = &$funcref($ax);
  my($fb) = &$funcref($bx);
  if ($fb > $fa) {
    # Switch roles of a and b so that we can go downhill in the direction
    # from a to b.
    ($ax, $bx) = ($bx, $ax);
    ($fa, $fb) = ($fb, $fa);
  }

  my($cx) = $bx + $GOLD*($bx-$ax);   # First guess for c.
  my($fc) = &$funcref($cx);

  # Keep looping here until we bracket.
  while ($fb >= $fc && $cx != $inf && $cx != -$inf) {   # JME: added the inf tests, and changed >= to > to make sure we keep searching all the way to inf if necessary in order to get $ax $bx $cx strictly in order
    # print("ax $ax bx $bx cx $cx // fa $fa fb $fb fc $fc\n"),

    # Compute u by parabolic extrapolation from a, b, c.
    # $TINY is used to prevent any possible division by zero.
    my($r) = ($bx-$ax)*($fb-$fc);
    my($q) = ($bx-$cx)*($fb-$fa);
    my($u) = $bx -(($bx-$cx)*$q - ($bx-$ax)*$r)/(2.0*&SIGN(&FMAX(abs($q-$r),$TINY),$q-$r));
    my($ulim) = $bx + $GLIMIT*($cx-$bx);
    my($fu);
    # We won't go farther than this.  Test various possibilities:
    if (($bx - $u)*($u - $cx) > 0) {    # Parabolic u is (strictly) between b and c: try it.
      ($fu) = &$funcref($u);
      if ($fu < $fc) {                  # Got a minimum between b and c.
        ($ax,$bx) = ($bx,$u);
        ($fa,$fb) = ($fb,$fu);
        return($ax, $bx, $cx, $fa, $fb, $fc) if ($ax-$bx)*($bx-$cx)>0 && $fb < $fa && $fb < $fc;
        die "mnbrak: oops, trying to return $ax $bx $cx out of order, or else middle value of $fa $fb $fc is not smallest\n";
      } elsif ($fu > $fb) {             # Got a minimum between a and u.
        $cx = $u;
        $fc = $fu;
        return($ax, $bx, $cx, $fa, $fb, $fc) if ($ax-$bx)*($bx-$cx)>0 && $fb < $fa && $fb < $fc;
        die "mnbrak: oops, trying to return $ax $bx $cx out of order, or else middle value of $fa $fb $fc is not smallest\n";
      }
      $u = $cx + $GOLD*($cx-$bx);       # Parabolic fit was no use.  Use default magnification.
      ($fu) = &$funcref($u);
    } elsif (($cx-$u)*($u-$ulim) > 0) {  # Parabolic fit is between c and its allowed limit
      ($fu) = &$funcref($u);
      if ($fu < $fc) {
        ($bx, $cx, $u) = ($cx, $u, $u+$GOLD*($u-$cx));  # JME: formerly $cx+$GOLD*($cx-$bx), but that seems to have been a bug since the new u might not be beyond the new cx.
        ($fb, $fc, $fu) = ($fc, $fu, &$funcref($u));
      }
    } elsif (($u-$ulim)*($ulim-$cx) > 0) {  # Limit parabolic u to maximum allowed value.  JME: Changed >= to > so that we are guaranteed $u > $cx strictly.  See comment at top of loop.
      $u=$ulim;
      ($fu) = &$funcref($u);
    } else {                            # Reject parabolic u, use default magnification.
      $u=$cx+$GOLD*($cx-$bx);
      ($fu)=&$funcref($u);
    }
    ($ax,$bx,$cx) = ($bx,$cx,$u);       # Eliminate oldest point and continue.
    ($fa,$fb,$fc) = ($fb,$fc,$fu);
  }
  return($ax, $bx, $cx, $fa, $fb, $fc) if ($ax-$bx)*($bx-$cx)>0  && $fb <= $fa && ($fb <= $fc || $cx==$inf || $cx==-$inf);
  die "mnbrak: oops, trying to return $ax $bx $cx out of order, or else middle value of $fa $fb $fc is not smallest but we didn't run into infinity with cx=$fc\n";
}


# Using the Van Wijngaarden-Dekker-Brent method, find the root of a
# function f (referenced by $funcref) between x1 and x2, where f(x1)
# and f(x2) must have different signs.  The root will be refined until
# its accuracy is $tol (which defaults to the machine epsilon if
# omitted).
#
# See easyzbrent for a sometimes easier way to call this.

sub zbrent {
  my($funcref, $x1, $x2, $tol) = @_;
  $tol = $machine_epsilon unless defined $tol;

  my($ITMAX) = 100;             # The maximum allowed number of iterations.
  my($EPS) = $machine_epsilon;  # Machine floating-point precision.  (Defined as 3.0e-8 in C version.)

  my($a,$b,$c)=($x1,$x2,$x2);
  my($d,$e,$min1,$min2);
  my($fa,$fb) = (&$funcref($a), &$funcref($b));
  my($p,$q,$r,$s,$tol1,$xm);

  die "zbrent: root must be bracketed between x1=$x1 and x2=$x2, but f(x1)=$fa, f(x2)=$fb" if $fb*$fa > 0;

  my($fc)=$fb;
  my($iter);
  for ($iter=1;$iter<=$ITMAX;$iter++) {
    if ($fb*$fc > 0) {
      $c=$a;                         # Rename a, b, c and adjust bounding interval d.
      $fc=$fa;
      $e=$d=$b-$a;
    }
    if (abs($fc) < abs($fb)) {
      $a=$b;
      $b=$c;
      $c=$a;
      $fa=$fb;
      $fb=$fc;
      $fc=$fa;
    }
    $tol1=2*$EPS*abs($b)+0.5*$tol;   # Convergence check.
    $xm=0.5*($c-$b);
    return $b if (abs($xm) <= $tol1 || $fb == 0);
    if (abs($e) >= $tol1 && abs($fa) > abs($fb)) {
      $s=$fb/$fa;                    # Attempt inverse quadratic interpolation.
      if ($a == $c) {
        $p=2*$xm*$s;
        $q=1-$s;
      } else {
        $q=$fa/$fc;
        $r=$fb/$fc;
        $p=$s*(2*$xm*$q*($q-$r)-($b-$a)*($r-1));
        $q=($q-1)*($r-1)*($s-1);
      }
      $q = -$q if ($p > 0);          # Check whether in bounds.
      $p=abs($p);
      $min1=3*$xm*$q-abs($tol1*$q);
      $min2=abs($e*$q);
      if (2*$p < ($min1 < $min2 ? $min1 : $min2)) {
        $e=$d;                       # Accept interpolation.
        $d=$p/$q;
      } else {
        $d=$xm;                      # Interpolation failed, use bisection.
        $e=$d;
      }
    } else {                         # Bounds decreasing too slowly, use bisection.
      $d=$xm;
      $e=$d;
    }
    $a=$b;                           # Move last best guess to $a.
    $fa=$fb;
    if (abs($d) > $tol1) {           # Evaluate new trial root.
      $b += $d;
    } else {
      $b += ($xm > 0 ? abs($tol1) : -abs($tol1));
    }
    $fb=&$funcref($b);
  }
  die "$0: zbrent: Maximum number of iterations ($ITMAX) exceeded";
}

1;
