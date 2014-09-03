CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = `$(CC) -dumpversion | awk '{if(NR==1 && $$1>="4.6") print "-lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -std=c99 -g"; else print "-lm -pthread -O2 -march=native -Wall -funroll-loops -std=c99 -g";}'`

all: rnnlm

rnnlm : rnnlm.c
	$(CC) rnnlm.c -o rnnlm $(CFLAGS)

clean:
	rm -rf rnnlm
