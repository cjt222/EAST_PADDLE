CXXFLAGS = -I include  -std=c++11 -O3 -I/root/anaconda2/envs/rrpn/include/python3.7m -I/root/anaconda2/envs/rrpn/include/python3.7m  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -O3 -ffunction-sections -pipe  -fdebug-prefix-map=/tmp/build/80754af9/python_1572016129546/work=/usr/local/src/conda/python-3.7.5 -fdebug-prefix-map=/root/anaconda2/envs/rrpn=/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall

LDFLAGS = $(shell python3-config --ldflags)

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
