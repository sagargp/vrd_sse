all:
	g++ vrd.cpp -g -o vrd -std=c++0x -I/usr/local/include -I/home/sagar/workspace/nrt/include -L/home/sagar/workspace/nrt/build -lnrtCore -lnrtImageProc -lboost_thread -lboost_serialization
	
test:
	g++ test.cpp -g -o test -std=c++0x -I/usr/local/include -I/home/sagar/workspace/nrt/include -L/home/sagar/workspace/nrt/build -lnrtCore -lnrtImageProc -lboost_thread -lboost_serialization

clean:
	rm -f vrd test
