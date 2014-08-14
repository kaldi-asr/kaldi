CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -std=c99 -g

all: rnnlm

rnnlm : rnnlm.c
	$(CC) rnnlm.c -o rnnlm $(CFLAGS)

clean:
	rm -rf rnnlm