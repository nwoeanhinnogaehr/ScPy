CXX := g++
CXXFLAGS := -I/usr/include/SuperCollider/plugin_interface \
	-I/usr/include/SuperCollider/common \
	`pkg-config --cflags --libs python3` \
	-fPIC
SC_EXT_DIR := ~/.local/share/SuperCollider/Extensions
OBJ_FILES := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

all : fsm.so

fsm.so : $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) *.o -shared -o fsm.so

%.o : %.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o $@

install : fsm.so fsm.sc
	cp -f fsm.sc fsm.so $(SC_EXT_DIR)

clean :
	rm -rf *.so *.o
