# Rules that enable valgrind debugging ("make valgrind")

valgrind: .valgrind

.valgrind:
	echo -n > valgrind.out
	for x in $(TESTFILES); do echo $$x>>valgrind.out; valgrind ./$$x >/dev/null 2>> valgrind.out; done
	! ( grep 'ERROR SUMMARY' valgrind.out | grep -v '0 errors' )
	! ( grep 'definitely lost' valgrind.out | grep -v -w 0 )
	rm valgrind.out
	touch .valgrind


# Rules that enable directory locks for parallel compilation
.PHONY: lock-dir
lock-dir:
	@if [[ "$(RECURSIVE)" != "true" && -d .lock ]]; then rmdir $(shell pwd)/.lock; fi
	@if [ -d .lock ]; then echo "[WAIT $(shell pwd) IS LOCKED]"; echo "(or unlock manually by 'rmdir $(shell pwd)/.lock')"; fi
	@while ! mkdir .lock 2>/dev/null; do sleep 1; done
	@echo "[LOCKED $(shell pwd)]"

unlock-dir=@rmdir .lock; echo "[UNLOCKED $(shell pwd)]"



