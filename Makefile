SOURCES=$(wildcard src/*.cpp)
HEADER=$(wildcard src/*.h)
MODULE=tree
SUFFIX=$(shell python3-config --extension-suffix)
TARGET=$(MODULE)$(SUFFIX)
INCLUDES=$(shell (python3 -m pybind11 --includes))
LDFLAGS=-O3 -Wall -shared -std=c++11 -fPIC

CC=g++

all: $(TARGET)


$(TARGET): $(SOURCES) $(HEADER)
	$(CC) -o $(TARGET) $(LDFLAGS) $(INCLUDES) $(SOURCES) 


.PHONY: clean
clean:
	if ls *$(SUFFIX) > /dev/null 2>&1; then rm *$(SUFFIX); fi