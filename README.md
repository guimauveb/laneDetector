# C++ / OpenCV Lane Detection program

**Basic road lane detection program inspired by various project I found on the web, most of them written in Python.**
**Works with images and videos sources - the brighter / higher quality the source is the better the program works** 
**No ML / AI involved.**

## Installation
1. Make sure you have OpenCV's latest version installed
2. I use pkg-config to link all opencv librairies but you can use another method.
3. If you use pkg-config copy the following command to compile the program:
    - clang++ laneDet.cpp -o laneDet `pkg-config --cflags --libs opencv4` -std=c++17
