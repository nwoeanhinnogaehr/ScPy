SC_EXT_DIR ?= ~/.local/share/SuperCollider/Extensions
CXX ?= g++
CXXFLAGS += -I/usr/include/SuperCollider/plugin_interface \
	-I/usr/include/SuperCollider/common \
	-I. \
	`python3-config --cflags` \
	-fPIC \
	-Wall -Wextra -std=c++11 \
	-D ABS_SOURCE_PATH=\"`pwd`\" \
	-g
LDFLAGS += `python3-config --ldflags` \
		   -shared
OBJ_FILES := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

all : scpy.so

scpy.so : $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) *.o -o scpy.so $(LDFLAGS)

%.o : %.cpp *.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

install : scpy.so scpy.sc
	cp -rf scpy.sc scpy.so SystemOverwrites $(SC_EXT_DIR)

clean :
	rm -rf *.so *.o
