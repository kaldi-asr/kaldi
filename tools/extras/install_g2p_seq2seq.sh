if [ ! -e g2p-seq2seq ];then
  git clone https://github.com/cmusphinx/g2p-seq2seq.git
  cd g2p-seq2seq/
  python setup.py install
fi
