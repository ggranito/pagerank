#
# To build with a different compiler / on a different platform, use
#     make PLATFORM=xxx
#
# where xxx is
#     icc = Intel compilers
#     gcc = GNU compilers
#     clang = Clang compiler (OS X default)
#
# Or create a Makefile.in.xxx of your own!
#

PLATFORM=icc
include Makefile.in.$(PLATFORM)

.PHONY: exe clean realclean


# === Executables

exe: pagerank.x

pagerank.x: pagerank.o mt19937p.o
	$(CC) $(OMP_CFLAGS) $^ -o $@

pagerank.o: pagerank.c
	$(CC) -c $(OMP_CFLAGS) $<

pagerank-mpi.x: pagerank-mpi.o mt19937p.o
	$(MPICC) $(MPI_CFLAGS) $^ -o $@

pagerank-mpi.o: pagerank-mpi.c
	$(MPICC) -c $(MPI_CFLAGS) $<

%.o: %.c
	$(CC) -c $(CFLAGS) $<


# === Cleanup and tarball

clean:
	rm -f *.o* *.x

realclean: clean
	rm -f pagerank.x pagerank-mpi.x
