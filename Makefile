#默认情况下，L1是被开启的，-Xptxas -dlcm=cg可以用来禁用L1

#compile parameters

CC = g++
#NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G --ptxas-options=-v
NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G
#NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true -G -Xcompiler -rdynamic -lineinfo
CFLAGS = -g -c #-fprofile-arcs -ftest-coverage -coverage #-pg
EXEFLAG = -g #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2
#NVCC = nvcc -arch=sm_35 -lcudadevrt -rdc=true 
#CFLAGS = -c #-fprofile-arcs -ftest-coverage -coverage #-pg
#CFLAGS = -c -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg
#EXEFLAG = -O2 #-fprofile-arcs -ftest-coverage -coverage #-pg #-O2
# TODO: try -fno-builtin-strlen -funswitch-loops -finline-functions

#add -lreadline -ltermcap if using readline or objs contain readline
library = #-lgcov -coverage

objdir = ./objs/
objfile = $(objdir)Util.o $(objdir)IO.o $(objdir)Match.o $(objdir)Graph.o

all: run.exe

run.exe: $(objfile) main/run.cpp
	$(NVCC) $(EXEFLAG) -o run.exe main/run.cpp $(objfile)

$(objdir)Util.o: util/Util.cpp util/Util.h
	$(CC) $(CFLAGS) util/Util.cpp -o $(objdir)Util.o

$(objdir)Graph.o: graph/Graph.cpp graph/Graph.h
	$(CC) $(CFLAGS) graph/Graph.cpp -o $(objdir)Graph.o

$(objdir)IO.o: io/IO.cpp io/IO.h
	$(CC) $(CFLAGS) io/IO.cpp -o $(objdir)IO.o

$(objdir)Match.o: match/Match.cu match/Match.h
	$(NVCC) $(CFLAGS) match/Match.cu -o $(objdir)Match.o

.PHONY: clean dist tarball test sumlines

clean:
	rm -f $(objdir)*
dist: clean
	rm -f *.txt *.exe

tarball:
	tar -czvf gsm.tar.gz main util match io graph Makefile README.md objs

test: main/test.o $(objfile)
	$(CC) $(EXEFLAG) -o test main/test.cpp $(objfile) $(library)

sumline:
	bash script/sumline.sh

