# clean and build
make clean
make all

# print command prompt
java -jar ChiUtf8Segmenter.jar

# example
java -jar ChiUtf8Segmenter.jar -mode5 example/test_sent.txt 186k_wordprobmap
mv example/test_sent.txt.seg example/test_sent.txt.seg0

# another example
java -jar ChiUtf8Segmenter.jar -mode5 example/test_sent.txt 186k_wordprobmap snumbers_u8.txt


