#!/bin/bash

# This file contains commands to create the FST examples.
export PATH=$PATH:~/kaldi/trunk/tools/openfst/bin/
# also need "dot" (from graphviz) on the command line.

mkdir -p figures
cat > symbols.txt <<EOF
<eps> 0 
a 1
b 2
c 3
d 4
e 5
f 6
g 7
x 8
y 9
z 10
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | dot -Tps > figures/acceptor.eps
0 1 a 0.0
1 2 b 0.0
2 0.0
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | dot -Tps > figures/acceptor_epsilon.eps
0 1 a 0.0
1 0 <eps> 0.0
1 0.0
EOF



cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | dot -Tps > figures/acceptor_nontriv.eps
0 1 a 0.0
1 1 a 0.0
1 2 b 0.0
2 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | dot -Tps > figures/acceptor_equiv.eps
0 1 a 0.0
0 0 a 0.0
1 2 b 0.0
2 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | dot -Tps > figures/acceptor_nondet.eps
0 1 a
1 2 b
0 3 a
3 4 c
2
4
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | dot -Tps > figures/acceptor_det.eps
0 1 a
1 2 b
1 3 c
2
3
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | dot -Tps > figures/acceptor_nonmin.eps
0 1 a
1 2 b
1 3 c
2 4 d
3 5 d
4
5
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | dot -Tps > figures/acceptor_min.eps
0 1 a
1 2 b
1 3 c
2 4 d
3 4 d
4
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one --acceptor | dot -Tps > figures/acceptor_weighted.eps
0 1 a 1.0
1 2 b 1.0
2 1.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor --show_weight_one | dot -Tps > figures/acceptor_costs.eps
0 3 a 0.0
0 1 b 2.0
1 2 c 1.0
2 1.0
3 0.0
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor  --show_weight_one | dot -Tps > figures/acceptor_notwin.eps
0 1 a 1.0
0 2 a 1.0
1 1 b 1.0
2 2 b 2.0
1 3 x 1.0
2 4 y 1.0
3 1.0
4 1.0
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor --show_weight_one | dot -Tps > figures/acceptor_sum.eps
0 1 a 0.0
0 2 a 1.0
1 1 b 1.0
1 4 d 1.0
2 3 b 2.0
1 2.0
3 2.0
4 1.0
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor --show_weight_one | dot -Tps > figures/acceptor_equiv_a.eps
0 1 a 0.0
1 1 b 1.0
1 1.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --acceptor --show_weight_one | dot -Tps > figures/acceptor_equiv_b.eps
0 1 a 1.0
1 1 b 1.0
1 0.0
EOF


# example showing mistake in twins definition.
# cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt --acceptor | fstdeterminize | fstprint --isymbols=symbols.txt --osymbols=symbols.txt --acceptor

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer.eps
0 1 a x 1.0
1 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_equiv.eps
0 1 a   <eps> 1.0
1 2 <eps> x   0.0
2 0.0
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_functional.eps
0 1 a  x 1.0
1 2 <eps> y 1.0
2 0.0
0 3 b z 1.0
3 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_nonfunctional.eps
0 1 a  x 1.0
1 2 <eps> y 1.0
2 0.0
0 3 a z 1.0
3 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_inv_a.eps
0 1 a  x  1.0
1 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_inv_b.eps
0 1 x  a  1.0
1 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_project_a.eps
0 1 a  x  1.0
0 1 b  y  1.0
1 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_project_b.eps
0 1 a  a  1.0
0 1 b  b  1.0
1 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_compose_a.eps
0 1 a  x  1.0
0 1 b  x  1.0
1 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_compose_b.eps
0 1 x y  1.0
1 0.0
EOF

cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_compose_c.eps
0 1 a  y  2.0
0 1 b  y  2.0
1 0.0
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_matrix.eps
0 1 a  x  2.0
1 2 b  y  2.0
1 2 b  <eps>  3.0
1 0.0
2 0.0
EOF


cat <<EOF | fstcompile --isymbols=symbols.txt --osymbols=symbols.txt | fstdraw --isymbols=symbols.txt --osymbols=symbols.txt --show_weight_one | dot -Tps > figures/transducer_deterministic.eps
0 1 a  x  2.0
0 1 b  x  2.0
1 2 c  y  2.0
1 3 d <eps> 2.0
EOF


cat <<EOF | fstcompile --acceptor | fstdraw --acceptor --show_weight_one | dot -Tps > figures/acceptor_utterance.eps
0 1 1 4.86
0 1 2 4.94
0 1 3 5.31
0 1 4 5.91
1 2 1 4.16
1 2 2 5.44
1 2 3 6.31
1 2 4 5.02
2 3 1 6.02
2 3 2 6.47
2 3 3 5.16
2 3 4 8.53
3 0.0
EOF

