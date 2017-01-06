#!/usr/bin/perl

# $Id: numhack.perl,v 1.4 1996/08/23 05:12:27 robertm Rel $
# preprocessor for numproc, potentially specialized for Broadcast News material

# tries to patch numproc's problems with:
#	- telephone numbers
#	- zip codes
# for example:
#   1-800-555-1212
#     =>  one - eight hundred -  five five five -  one two one two
#   (215) 555-1212
#     =>  two one five -  five five five -  one two one two
#   212/285-9400
#     =>  two one two -  two eight five -  nine four zero zero
#   1-(800)-CAR-CASH
#     =>  one - eight hundred -CAR-CASH
#   New York, NY 10007
#     =>  New York, NY  one zero zero zero seven
#   Philadelphia, PA 19104-6789
#     =>  Philadelphia, PA  one nine one oh four -  six seven eight nine

# may leave behind extra spaces here and there, but later processes ought
# to correct that...

@ones_oh=("oh","one","two","three","four",
	  "five","six","seven","eight","nine");

while(<>)
{
    next unless /\d/;		# skip lines without numbers
    next if /^<\/?[aps]/;	# skip SGML

    # probable Zip codes
    s/\b(\d{5}-\d{4})\b/&SpellDigits($1)/eg;	# 12345
    s/\b(\d{5})\b/&SpellDigits($1)/eg;		# 12345-6789

    # phone numbers
    s=(^| )([1l][- ])?\(?([2-9]\d{2})\)?[-/]? ?(\d{3})-(\d{4})\b=&SpellTel($2,$3,$4,$5)=eg; # 215-555-1212 etc.
    s/(^| )(\d{3}-\d{4})\b/&SpellDigits($2)/eg;	# 555-1212
    s/\b1-\(?800\)?(\W)/ one - eight hundred $1/g;	# isolated 1-800
    s/([Aa]rea code) (\d{3})(\W)/"$1 ".&SpellDigits($2)."$3"/eg;

} continue {
    print;
}

exit;

sub SpellDigits
{
    local($num)=$_[0];
    $num =~ s/(\d)(\D)(\d)/$1 $2 $3/g; # add space around non-digits
    # isolated zeros become "oh", string of them become "zero ..."
    $num =~ s/(00+)/" zero" x length($1)/eg;
    $num =~ s/(\d)/" $ones_oh[$1]"/eg;
    return $num;
}

sub SpellTel
{
    local($pre,$area,$exch,$rest)=@_;
    $return = $pre ? " one -" : " ";
    if ($area =~ /(\d)00/)
    {
	$return .= &SpellDigits($1);
	$return .= " hundred";
    }
    else
    {
	$return .= &SpellDigits($area);
    }
    $return .= " - ";

    $return .= &SpellDigits($exch);
    $return .= " - ";
    $return .= &SpellDigits($rest);

    return $return;
}
