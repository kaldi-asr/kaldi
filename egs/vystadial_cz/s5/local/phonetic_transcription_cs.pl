#!/usr/bin/perl

use strict;
use warnings;
use utf8;
use Encode;

# $ PhoneticTranscriptionCS.pl [inputFile inputFile2 ...] outputFile
#
# Converts Czech text in CAPITALS in utf8 to Czech phonetic alphabet in
# utf8. All input files will be concatenated into the output file. If no
# input files are specified, reads from STDIN.
#
# If you want the script to operate in another encoding, set the EV_encoding
# environment variable to the desired encoding.
#
# This is a rewrite of "vyslov" shell-script by Nino Peterek and Jan Oldrich Kruza, which was using tools
# written by Pavel Ircing. These are copy-pasted including comments into this
# script.

my $enc = 'utf8';

my $out_fn = pop @ARGV;
if ($out_fn) {
    close STDOUT;
    open STDOUT, '>', $out_fn or die "Couldn't open '$out_fn': $!";
}

my %seen = ();
while (<>) {
    for (decode($enc, $_)) {
#        if (/[^\w\s]/) {
#            chomp;
#            print encode($enc, $_), (' ' x 7), "sp\n";
#            next
#        }
        chomp;
        $_ = uc($_);

        print encode($enc, $_);
        print(' ' x 7);
        exceptions();
        transcription();
        tr/[A-Z]/[a-z]/;
        prague2pilsen();
        infreq();

#        while ($_ =~ /(.)/g) {
#            $seen{$1}++;
#        }

        print encode($enc, $_);
        print "\n";
    }
}

#print "unique chars are: ";
#foreach (sort(keys %seen)) {
#    print encode($enc, $_);
#}

sub exceptions {
    s/AA/A/g;

    s/AKTI/AKTY/g;
    s/ANTI/ANTY/g;
    s/ARKTI/ARKTY/g;
    s/ATIK/ATYK/g;
    s/ATRAKTI/ATRAKTY/g;
    s/AUDI/AUDY/g;
    s/AUTOMATI/AUTOMATY/g;
    s/^BARRANDOV/BARANDOV/g;
    s/CAUSA/KAUZA/g;
    s/CELSIA/CELZIA/g;
    s/^CHAPLIN/ČAPLIN/g;
    s/CHIL/ČIL/g;
    s/DANIH/DANYH/g;
    s/DEALER/D ii LER/g;
    s/DIAG/DYAG/g;
    s/DIET/DYET/g;
    s/DIF/DYF/g;
    s/DIG/DYG/g;
    s/DIKT/DYKT/g;
    s/DILET/DYLET/g;
    s/DIPL/DYPL/g;
    s/DIRIG/DYRYG/g;
    s/DISK/DYSK/g;
    s/DISP/DYSP/g;
    s/DISPLAY/DYSPLEJ/g;
    s/DIST/DYST/g;
    s/DIVIDE/DYVIDE/g;
    s/DUKTI/DUKTY/g;
    s/EDIC/EDYC/g;
    s/EFEKTIV/EFEKTYV/g;
    s/ELEKTRONI/ELEKTRONY/g;
    s/ENERGETIK/ENERGETYK/g;
    s/ERROR/EROR/g;
    s/ETIK/ETYK/g;
    s/^EX([AEIOUÁÉÍÓÚŮ])/EGZ$1/g;
    s/FEMINI/FEMINY/g;
    s/FINIŠ/FINYŠ/g;
    s/FINITI/FINYTY/g;
    s/GATIV/GATYV/g;
    s/GENETI/GENETY/g;
    s/GIENI/GIENY/g;
    s/GITI/GITY/g;
    s/^GOETH/GÉT/g;
    s/IMUNI/IMUNY/g;
    s/INDIV/INDYV/g;
    s/ING/YNG/g;
    s/INICI/INYCI/g;
    s/INVESTI/INVESTY/g;
    s/KANDI/KANDY/g;
    s/KARATI/KARATY/g;
    s/KARDI/KARDY/g;
    s/KLAUS/KLAUZ/g;
    s/KOMODIT/KOMODYT/g;
    s/KOMUNI/KOMUNY/g;
    s/KONDI/KONDY/g;
    s/KONSOR/KONZOR/g;
    s/KREDIT/KREDYT/g;
    s/KRITI/KRITY/g;
    s/LEASING/L ii z ING/g;
    s/MANAG/MENEDŽ/g;
    s/MANIP/MANYP/g;
    s/MATI/MATY/g;
    s/MEDI/MEDY/g;
    s/MINI/MINY/g;
    s/MINUS/MÝNUS/g;
    s/MODERNI/MODERNY/g;
    s/MONIE/MONYE/g;
    s/MOTIV/MOTYV/g;
    s/^MOZART/MÓCART/g;
    s/^NE/NE!/g;
    s/^NEWTON/ŇŮTN/g;
    s/NIE/NYE/g;
    s/NII/NYY/g;
    s/NJ/Ň/g;
    s/NSTI/NSTY/g;
    s/^ODD/OD!D/g;
    s/^ODI(?=[^V])/ODY/g;
    s/^ODT/OT!T/g;
    s/OPTIM/OPTYM/g;
    s/ORGANI/ORGANY/g;
    s/^PANASONIC/PANASONYK/g;
    s/PANICK/PANYCK/g;
    s/^Patton/PETN/g;
    s/PEDIATR/PEDYATR/g;
    s/PERVITI/PERVITY/g;
    s/^PODD/POD!D/g;
    s/^PODT/POT!T/g;
    s/POLITI/POLITY/g;
    s/^POULI/PO!ULI/g;
    s/POZIT/POZYT/g;
    s/^PŘED(?=[^Ě])/PŘED!/g;
    s/PRIVATI/PRIVATY/g;
    s/PROSTITU/PROSTYTU/g;
    s/RADIK/RADYK/g;
    s/^RADIO/RADYO/g;
    s/^RÁDI(.)/RÁDY$1/g;
    s/RELATIV/RELATYV/g;
    s/RESTITU/RESTYTU/g;
    s/^ROCK/ROK/g;
    s/^ROZ/ROZ!/g;
    s/RUTIN/RUTYN/g;
    s/^SCHENGEN/ŠENGEN/g;
    s/^SEBE/SEBE!/g;
    s/SHOP/sz O P/g;
    s/^SHO/SCHO/g;
    s/SOFTWAR/SOFTVER/g;
    s/SORTIM/SORTYM/g;
    s/SPEKTIV/SPEKTYV/g;
    s/STATISTI/STATYSTY/g;
    s/STIK/STYK/g;
    s/STIMUL/STYMUL/g;
    s/^STROSSMAYER/ŠTROSMAJER/g;
    s/STUDI/STUDY/g;
    s/SUPERLATIV/SUPERLATYV/g;
    s/TECHNI/TECHNY/g;
    s/TELECOM/TELEKOM/g;
    s/TELEFONI/TELEFONY/g;
    s/TEMATI/TEMATY/g;
    s/^TESCO/TESKO/g;
    s/TETIK/TETYK/g;
    s/TEXTIL/TEXTYL/g;
    s/TIBET/TYBET/g;
    s/TIBOR/TYBOR/g;
    s/TICK/TYCK/g;
    s/TIRANY/TYRANY/g;
    s/TITUL/TYTUL/g;
    s/TRADI/TRADY/g;
    s/UNIVER/UNYVER/g;
    s/VENTI/VENTY/g;
    s/VERTIK/VERTYK/g;
    s/^WAGNER/WÁGNER/g;
    s/^WATT/VAT/g;
    s/^WEBBER/VEBER/g;
    s/^WEBER/VEBER/g;
    s/^WILSON/VILSON/g;

}

sub transcription {
    # namapování nechtěných znaků na model ticha
    s/^.*[0-9].*$/sil/g;

    # náhrada víceznakových fonémů speciálním znakem, případně rozepsání znaku na více fonémů
    s/CH/#/g;
    s/W/V/g;
    s/Q/KV/g;
    s/DŽ/&/g;  # v původním vyslov nefungovalo
    s/DZ/@/g;
    s/X/KS/g;

    # ošetření Ě
    s/([BPFV])Ě/$1JE/g;
    s/DĚ/ĎE/g;
    s/TĚ/ŤE/g;
    s/NĚ/ŇE/g;
    s/MĚ/MŇE/g;
    s/Ě/E/g;

    # změkčující i
    s/DI/ĎI/g;
    s/TI/ŤI/g;
    s/NI/ŇI/g;
    s/DÍ/ĎÍ/g;
    s/TÍ/ŤÍ/g;
    s/NÍ/ŇÍ/g;

    # asimilace znělosti
    s/B$/P/g;
    s/B([PTŤKSŠCČ#F])/P$1/g;
    s/B([BDĎGZŽ@&H])$/P$1/g;
    s/P([BDĎGZŽ@&H])/B$1/g;
    s/D$/T/g;
    s/D([PTŤKSŠCČ#F])/T$1/g;
    s/D([BDĎGZŽ@&H])$/T$1/g;
    s/T([BDĎGZŽ@&H])/D$1/g;
    s/Ď$/Ť/g;
    s/Ď([PTŤKSŠCČ#F])/Ť$1/g;
    s/Ď([BDĎGZŽ@&H])$/Ť$1/g;
    s/Ť([BDĎGZŽ@&H])/Ď$1/g;
    s/V$/F/g;
    s/V([PTŤKSŠCČ#F])/F$1/g;
    s/V([BDĎGZŽ@&H])$/F$1/g;
    s/F([BDĎGZŽ@&H])/V$1/g;
    s/G$/K/g;
    s/G([PTŤKSŠCČ#F])/K$1/g;
    s/G([BDĎGZŽ@&H])$/K$1/g;
    s/K([BDĎGZŽ@&H])/G$1/g;
    s/Z$/S/g;
    s/Z([PTŤKSŠCČ#F])/S$1/g;
    s/Z([BDĎGZŽ@&H])$/S$1/g;
    s/S([BDĎGZŽ@&H])/Z$1/g;
    s/Ž$/Š/g;
    s/Ž([PTŤKSŠCČ#F])/Š$1/g;
    s/Ž([BDĎGZŽ@&H])$/Š$1/g;
    s/Š([BDĎGZŽ@&H])/Ž$1/g;
    s/H$/#/g;
    s/H([PTŤKSŠCČ#F])/#$1/g;
    s/H([BDĎGZŽ@&H])$/#$1/g;
    s/#([BDĎGZŽ@&H])/H$1/g;
    s/\@$/C/g;
    s/\@([PTŤKSŠCČ#F])/C$1/g;
    s/\@([BDĎGZŽ@&H])$/C$1/g;
    s/C([BDĎGZŽ@&H])/\@$1/g;
    s/&$/Č/g;
    s/&([PTŤKSŠCČ#F])/Č$1/g;
    s/&([BDĎGZŽ@&H])$/Č$1/g;
    s/Č([BDĎGZŽ@&H])/&$1/g;
    s/Ř$/>/g;
    s/Ř([PTŤKSŠCČ#F])/>$1/g;
    s/Ř([BDĎGZŽ@&H])$/>$1/g;
    s/([PTŤKSŠCČ#F])Ř/$1>/g;


    #zbytek
    s/NK/ng K/g;
    s/NG/ng G/g;
    s/MV/mg V/g;
    s/MF/mg F/g;
    s/NŤ/ŇŤ/g;
    s/NĎ/ŇĎ/g;
    s/NŇ/Ň/g;
    s/CC/C/g;
    s/DD/D/g;
    s/JJ/J/g;
    s/KK/K/g;
    s/LL/L/g;
    s/NN/N/g;
    s/MM/M/g;
    s/SS/S/g;
    s/TT/T/g;
    s/ZZ/Z/g;
    s/ČČ/Č/g;
    s/ŠŠ/Š/g;
    s/-//g;

    # závěrečný přepis na HTK abecedu
    s/>/rsz /g;
    s/EU/eu /g;
    s/AU/au /g;
    s/OU/ou /g;
    s/Á/aa /g;
    s/Č/cz /g;
    s/Ď/dj /g;
    s/É/ee /g;
    s/Í/ii /g;
    s/Ň/nj /g;
    s/Ó/oo /g;
    s/Ř/rzs /g;
    s/Š/sz /g;
    s/Ť/tj /g;
    s/Ú/uu /g;
    s/Ů/uu /g;
    s/Ý/ii /g;
    s/Ž/zs /g;
    s/Y/i /g;
    s/&/dzs /g;
    s/\@/ts /g;
    s/#/ch /g;
    s/!//g;
    s/([A-Z])/$1 /g;

    # crazy characters mapped to closest phones
    s/Ü/uu /g;
    s/Ö/o /g;
    s/Ć/ch /g;
    s/Ľ/l /g;
    s/Ś/sz /g;
    s/Ű/uu /g;
    s/Ź/zs /g;
    s/Ń/nj /g;
    s/Ę/e /g;
    s/Ě/e /g;
    s/Ĺ/l /g;
    s/Ľ/l /g;
    s/Ł/l /g;
    s/Â/a /g;
    s/Ä/a /g;
    s/Ç/c /g;
    s/Ë/e /g;
    s/Î/i /g;
    s/Ô/o /g;
    s/Ő/o /g;

#    s/$/ sp/g;
}

sub prague2pilsen {
    s/au/aw/g;
    s/ch/x/g;
    s/cz/ch/g;
    s/dzs/dzh/g;
    s/es/e s/g;
    s/eu/ew/g;
    s/ou/ow/g;
    s/rsz/rsh/g;
    s/rzs/rzh/g;
    s/sz/sh/g;
    s/ts/dz/g;
    s/zs/zh/g;
}

sub infreq {
    s/dz/c/g;
    s/dzh/ch/g;
    s/ew/e u/g;
    s/mg/m/g;
    s/oo/o/g;
}


