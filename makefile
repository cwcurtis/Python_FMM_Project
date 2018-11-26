CC = gcc
CYTHON = cython
CFLAGS = -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing
CPPFLAGS = -I/usr/include/python2.7 

TARGETS = fmm_tree_c.so list_build_routines_c.so
CYTHON_SOURCE = fmm_tree_c.pyx list_build_routines_c.pyx
CYTHON_SECONDARY = $(CYTHON_SOURCE:.pyx=.c) $(TARGETS:=.c)

all: $(TARGETS)

%.so: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

%.c: %.pyx
	$(CYTHON) $^

clean:
	$(RM) *.so $(CYTHON_SECONDARY) $(TARGETS)
