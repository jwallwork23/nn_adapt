all: mesh

TEST_CASES	= 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

mesh:
	for test_case in $(TEST_CASES); do \
		gmsh -2 $$test_case.geo; \
	done
