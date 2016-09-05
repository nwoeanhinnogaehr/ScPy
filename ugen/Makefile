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
	$(CXX) *.o -shared -o scpy.so $(CXXFLAGS) $(LDFLAGS) 

%.o : %.cpp *.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

install : scpy.so scpy.sc
	cp -rf scpy.sc scpy.so SystemOverwrites $(SC_EXT_DIR)

ubuntuinstall: scpy.so
	cd loadit && make install

clean :
	rm -rf *.so *.o loadit/*.{so,o}
	rm -rf ../py/__pycache__
