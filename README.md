# C++ / OpenCV Basic Lane Detection 
![Lane Detector](/imgs/screenshot.jpg)
**Basic road lane detection program inspired by various project I found on the web, most of them written in Python.**
**Supports images and videos sources**
**It works well with good quality sources (I would say somewhere around at least YouTube 720p quality).**
**The higher the contrast between the lanes and the road the easier it is for the program to correctly detect lanes location.**

**No ML / AI involved.**

## Installation
1. Make sure you have OpenCV's latest version installed
2. I use pkg-config to link all opencv librairies but you can use another method.
3. If you use pkg-config copy the following command to compile the program:
```
    - clang++ laneDet.cpp -o laneDet `pkg-config --cflags --libs opencv4` -std=c++17
```
