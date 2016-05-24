INCLUDES := -I/usr/include/SuperCollider/plugin_interface -I/usr/include/SuperCollider/common
SC_EXT_DIR := ~/.local/share/SuperCollider/Extensions

all : fsm.so

fsm.so : fsm.cpp
	g++ $(INCLUDES) fsm.cpp -shared -fPIC -o fsm.so

install : fsm.so fsm.sc
	cp -f fsm.sc fsm.so $(SC_EXT_DIR)
