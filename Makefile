vrd: vrd.cpp vrd_sse.o
	g++ vrd.cpp vrd_sse.o -g -o vrd -std=c++0x -I/usr/local/include -I/home/sagar/workspace/nrt/include -L/home/sagar/workspace/nrt/build -lnrtCore -lnrtImageProc -lboost_thread -lboost_serialization -msse -msse2 -msse3 -mmmx

vrd_sse.o: vrd_sse.h vrd_sse.cpp
	g++ vrd_sse.cpp -fPIC -O3 -g -msse -c -o vrd_sse.o

test:
	g++ test.cpp -g -o test -std=c++0x -I/usr/local/include -I/home/sagar/workspace/nrt/include -L/home/sagar/workspace/nrt/build -lnrtCore -lnrtImageProc -lboost_thread -lboost_serialization

mex: vrd_sse.o VRD.cpp
	mex VRD.cpp vrd_sse.o

clean:
	rm -f vrd test vrd_sse.o *.mex*
