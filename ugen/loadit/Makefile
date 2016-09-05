SC_EXT_DIR ?= ~/.local/share/SuperCollider/Extensions
CXX ?= clang++
CXXFLAGS += -I/usr/include/SuperCollider/plugin_interface \
	-I/usr/include/SuperCollider/common \
	-I. \
	`pkg-config --cflags python3` \
        `python3-config --cflags` \
        -DOURPYTHONLIB=\"`bash YOURPYTHONLIB.sh`\" \
	-fPIC \
	-Wall -Wextra -std=c++11 \
	-D ABS_SOURCE_PATH=\"`pwd`\" \
	-g
LDFLAGS += `pkg-config --libs python3` `python3-config --ldflags`
OBJ_FILES := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

all : scpy.so

scpy.so : $(OBJ_FILES) YOURPYTHONLIB.sh
	$(CXX) *.o -shared -o scpy.so $(CXXFLAGS) $(LDFLAGS) 

%.o : %.cpp *.h YOURPYTHONLIB.sh
	$(CXX) $(CXXFLAGS) -c $< -o $@

install : scpy.so scpy.sc
	cp -f scpy.sc scpy.so $(SC_EXT_DIR)
	cp -f ../scpy.so $(SC_EXT_DIR)/scpy.os

clean :
	rm -rf *.so *.o
