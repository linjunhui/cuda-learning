
# 编译器
CXX = g++

# 连接库
CXXFLAGS = `pkg-config --cflags --libs opencv4`
LDFLAGS = `pkg-config --libs opencv4`

TARGET = main
SCR = main.cpp

$(TARGET): $(SCR)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)