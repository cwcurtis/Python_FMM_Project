TARGETS = setup.py 
TESTS = tester_for_import.py
CYTHON_SOURCE = fmm_tree_c.pyx list_build_routines_c.pyx
CYTHON_SECONDARY = $(CYTHON_SOURCE:.pyx=.c) $(TARGETS:=.c)

all: $(TARGETS)
	python $(TARGETS) build_ext --inplace

profile: $(TARGETS)	
	kernprof -l $(TARGETS)
	python -m line_profiler setup.py.lprof
	python setup.py build_ext --inplace
	kernprof -l $(TESTS)
	python -m line_profiler tester_for_import.py.lprof > profiled_code.txt && gedit profiled_code.txt
clean:
	$(RM) *.so $(CYTHON_SECONDARY)
