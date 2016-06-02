CXX := clang++
CXXFLAGS := -I/usr/include/SuperCollider/plugin_interface \
	-I/usr/include/SuperCollider/common \
	-I. \
	`pkg-config --cflags python3` \
	-fPIC \
	-Wall -Wextra -std=c++11 \
	-D ABS_SOURCE_PATH=\"`pwd`\"
LDFLAGS := `pkg-config --libs python3`
SC_EXT_DIR := ~/.local/share/SuperCollider/Extensions
OBJ_FILES := $(patsubst %.cpp,%.o,$(wildcard *.cpp))

all : fsm.so

fsm.so : $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) *.o -shared -o fsm.so

%.o : %.cpp %.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

install : fsm.so fsm.sc
	cp -f fsm.sc fsm.so $(SC_EXT_DIR)

clean :
	rm -rf *.so *.o
