## 编译与执行
```
mkdir build
cd build
camek  ..
make
./OpencvDemo
```

| 属性           | 说明                 |
| ------------ | ------------------ |
| `rows`       | 图像的高度（像素行数）        |
| `cols`       | 图像的宽度（像素列数）        |
| `channels()` | 每个像素的通道数（如3）       |
| `type()`     | 图像数据类型（如 CV\_8UC3） |
| `data`       | 图像像素数据的原始指针        |

- 每个像素点有3个通道， BGR顺序存储
```
cv::Vec3b pixel = image.at<cv::Vec3b>(0, 0);
std::cout << "Top-left pixel BGR: "
          << (int)pixel[0] << ", "
          << (int)pixel[1] << ", "
          << (int)pixel[2] << std::endl;
```

## cmake


## Makefile
### 关于pkg-config命令
pkg-config  --cflags opencv4 ->  -I/usr/include/opencv4
pkg-config  --libs opencv4 ->  -lopencv_core -lopencv_flann -lopencv_imgproc -lopencv_intensity_transform -lopencv_ml -lopencv_phase_unwrapping -lopencv_photo -lopencv_plot -lopencv_reg -lopencv_surface_matching -lopencv_video -lopencv_videoio -lopencv_videost

## 关于反引号 ``
执行shell命令，获取结果作为变量的值
CXXFLAGS = `pkg-config --cflags opencv4`

```
# 编译器
CXX = g++

# 使用 pkg-config 获取 OpenCV 编译参数
CXXFLAGS = `pkg-config --cflags opencv4`
LDFLAGS  = `pkg-config --libs opencv4`

# 目标文件
TARGET = main
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)

```