/* Basic road lane detection program inspired by various project I found on the web, most of them written in 
 * Python. No ML / AI involved. 
 *
 *      TODO - Discard lines with an extreme slope (set the same limit as the ones 
 *      returned by Hough transform
 *
 *       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *       ! Have 3 or 4 differents methods of recognizing the lanes according to the results !
 *       ! and conditions (weather, light...). If no lane is detected after a certain time  !    
 *       ! switch to another method.                                                        !
 *       ! (adaptive thresholding (darker scenes), canny edge on brighter scenes, what else ?                                                                       !
 *       ! Define the ROI according to src and road type !! 
 *       ! Set min width between lines detected !(approx width of the lane according to src !
 *       ! and road)                                      !                                                                 !
 *       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *
 */
#define NDEBUG                 // Disable Boost debug mode
#define BRIGHTNESS_L    100
#define BRIGHTNESS_A    150
#define BRIGHTNESS_H    220

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>

#include "Poly/PolyFitBoost.hpp"
#include "Poly/Poly1d.hpp"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>

using namespace cv;

/* Command line arguments : 
 * -v [video]   : use video as input 
 * -i [image]   : use image as input
 * -f s | m | l : fov - small, medium, large (default is medium 
 * -- r render  : bool - if set, render image or video in a separate window
 */
std::string filename;
bool v   = false;
bool im  = false;
bool r   = false;
int f    = 2;

static int printUsage(const char *progName)
{
    std::cout
        << "------------------------------------------------------" << std::endl
        << "This program tries to detect road lanes from an image or a video." << std::endl
        << "------------------------------------------------------" << std::endl
        << "Usage: "<< progName << " -i [image] -v [video] -f [fov] --r [render]" << std::endl
        << std::endl;

    return 0;
}

static int argsParser(int argc, char **argv)
{
    if (argc  <= 2) {
        printUsage(argv[0]);
        return -1;
    }

    for (size_t i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-i")) {
            if (!argv[i+1]) {
                std::cerr << "Error. Couldn't parse argument." << std::endl;
                return -1;
            }
            else {
                filename = argv[i+1];
                im = true;
            }
        }
        if (!strcmp(argv[i], "-v")) {
            if (!argv[i+1]) {
                std::cerr << "Error. Couldn't parse argument." << std::endl;
                return -1;
            }
            else {
                filename = argv[i+1];
                v = true;
            }
        }
        if (!strcmp(argv[i], "--r")) {
            r = true;
        }
        if (!strcmp(argv[i], "-f")) {
            if (!argv[i+1]) {
                std::cerr << "Error. Couldn't parse argument." << std::endl;
            }
            else if (!strcmp(argv[i+1], "s")) {
                f = 1;
            }
            else if (!strcmp(argv[i+1], "m")) {
                ;
            }
            else if (!strcmp(argv[i+1], "l")){
                f = 3;
            }
        }
    }
    return 0;
}

// Mat setContrast(Mat& img);
// Mat considerW(Mat& img);
Mat denoise(Mat& img);
Mat adaptBinary(Mat& img);
int getBrightness(Mat& img);
Mat getTheROI(Mat& img, int fov);
Mat betterCanny(Mat& img, double median);
double medianMat(Mat img);
std::vector<Vec4i> HoughTransform(Mat& img, Mat& src);
int processImage(std::string filename, bool render, int fov);
int processVideo(std::string filename, bool render, int fov);
std::vector<int> getLinearFunction(std::vector<Vec4i> lines, int min_y, int max_y);
Mat drawLines(std::vector<int> finalLines, Mat& src);
Mat drawArea(std::vector<int> recPoints, Mat& src);

int main(int argc, char **argv)
{
    // TODO - Add a slider to adjust parameters 
    argsParser(argc, argv);
    if (im)
        processImage(filename, r, f);
    else if (v)
        processVideo(filename, r, f);
    return 0;
}

Mat denoise(Mat& img)
{
    /* Benchmark */
    double t;
    t = (double)getTickCount();
    Mat denoised;  
    // Fast (but bad) denoise 
    // blur(img, denoised, Size(11, 11));
    // Good (but expensive) denoise 
    // fastNlMeansDenoising(img, denoised, 3, 7, 21);
    // Median blur
    medianBlur(img, denoised, 7);
    // imshow("denoised", denoised);
    // waitKey();
    t = (double)1000*(getTickCount() - t) / getTickFrequency();
    //std::cout << "denoise took\t\t" << t << " milliseconds to process the image." << std::endl;
    return denoised;
}

Mat adaptBinary(Mat& img, int cond)
{   
    // denoise image
    // Best thresholding method according to the conditions
    Mat binary;
    // TODO - Best values ??
    adaptiveThreshold(img, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 9, 2);
    // imshow("binary", binary);
    // waitKey();

    return binary;
}


Mat getTheROI(Mat& img, int fov)
{
    /* Benchmark */
    double t;
    t = (double)getTickCount();
    // 1. Frame pre-processing
    // b. ROI selection
    /* The top half of the image is not likely to contain any road lanes.
     * The ROI will be the lower half of the image. */
    int width, height;
    height  = img.size().height;
    width   = img.size().width;

    Mat mask = Mat::zeros(img.size(), img.type());
    int lineType = LINE_8;

    Point poly_points[1][3];

    // TODO - Adjust fov precisely according to a type of road
    // Small FOV
    if (fov == 1) {
        poly_points[0][0] = Point((width * 1/8), height);
        poly_points[0][1] = Point(width / 2, height / 1.8);
        poly_points[0][2] = Point((width * 7/8), height);
    }
    // Medium FOV
    else if (fov == 2) {
        poly_points[0][0] = Point((width * 1/10), height);
        poly_points[0][1] = Point(width / 2, height / 1.8);
        poly_points[0][2] = Point((width * 9/10), height);
    }
    // Large FOV
    else if (fov == 3) {
        poly_points[0][0] = Point(0, height);
        poly_points[0][1] = Point(width / 2, height / 1.8);
        poly_points[0][2] = Point((width), height);
    }

    const Point* ppt[1] = {poly_points[0]};
    int npt[] = {3};

    fillPoly(mask,
            ppt,
            npt,
            1,
            Scalar(255 ,255 ,255),
            lineType);

    /* Apply the mask on the source image. */
    Mat dst;
    bitwise_and(img, mask, dst);

    /* Display mask */
    // TODO - add command line argument
    // imshow("roi", mask);
    // waitKey();

    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    //std::cout << "getTheROI took\t\t" << t << " milliseconds to process the image." 
    //   << std::endl;

    return dst;
}

Mat betterCanny(Mat& img, double median)
{
    /* Benchmark */
    double t;
    t = (double)getTickCount();
    /* Get the best result from the Canny edge detector by detecting the ideal lower and upper
     * threshold values. */
    // A sigma of 0.33 seems to yield the best results.
    double sigma = 0.33;
    double lower = std::max(0.0, ((1.0 - sigma) * median));
    double upper = std::min(255.0, ((1.0 + sigma) * median));

    Mat edged = Mat::zeros(img.size(), img.type());
    Canny(img, edged, lower, upper);

    // imshow("edged", edged);
    // waitKey();

    /* Benchmark */
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    //std::cout << "betterCanny took\t" << t << " milliseconds to process the image." 
    //   << std::endl;

    return edged;
}

double medianMat(Mat img)
{
    /* Benchmark */
    double t;
    t = (double)getTickCount();
    /* In order to get the optimal low and high threshold values for our Canny edge detector,
     * we need to compute the median pixel value of the image. */
    // Spread img to a single row
    img = img.reshape(0,1);
    std::vector<int> v;
    // Convert our Mat object to a vector
    img.copyTo(v);
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    int vn = v[n];

    /* Benchmark */
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    //std::cout << "medianMat took\t\t" << t << " milliseconds to process the image." 
    //   << std::endl;

    // If v is an odd sized vector, return the middle element
    if (v.size() % 2)
        return vn;
    // If it's an even sized vector, return the middle of the two middle elements
    else
        return 0.5*(vn+v[n-1]);
}

std::vector<Vec4i> HoughTransform(Mat& img)
{
    /* TODO  - Find the ideal parameters for Hough 
     * dst: Output of the edge detector. It should be a grayscale image 
     * (although in fact it is a binary one).
     * lines: A vector that will store the parameters (x_{start}, y_{start}, x_{end}, y_{end}) 
     * of the detected lines
     * rho : The resolution of the parameter r in pixels. We use 1 pixel.
     * theta: The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
     * threshold: The minimum number of intersections to “detect” a line
     * minLinLength: The minimum number of points that can form a line. Lines with less than 
     * this number of points are disregarded.
     * maxLineGap: The maximum gap between two points to be considered in the same line 
     */

    /* Benchmark */
    double t;
    t = (double)getTickCount();

    Mat houghed = img.clone();

    cvtColor(img, houghed, COLOR_GRAY2BGR);
    std::vector<Vec4i> lines;
    HoughLinesP(img, lines, 4, CV_PI/180, 30, 40, 10);

    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    //std::cout << "Hough transform took\t" << t << " milliseconds to process the image." 
    //   << std::endl;

    return lines;
}

std::vector<int> getLinearFunction(std::vector<Vec4i> lines, int min_y, int max_y)
{
    /* Benchmark */
    double t;
    t = (double)getTickCount();

    // /* Before grouping lines */
    // for (size_t i = 0; i < lines.size(); i++) {
    //     Vec4i l = lines[i];
    //     line(houghed, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
    // }

    /* Group left and right lines and consider only one representation for each group */
    /* We could have used vectors containing Vec2i vectors, but we need to pass a 1 dimension 
     * vector to the polyfit function. */
    std::vector<double> left_line_x;
    std::vector<double> left_line_y;
    std::vector<double> right_line_x;
    std::vector<double> right_line_y;

    double slope;

    //TODO - Maybe find a better way to pass values to vectors
    for (size_t i = 0; i < lines.size(); i++) {
        /* If x2 - x1 == 0, continue */
        if ((lines[i][2] - lines[i][0]) == 0)
            continue;
        slope = (double)(lines[i][3] - lines[i][1]) / (lines[i][2] - lines[i][0]);
        /* Only consider (relatively) extreme slopes - adjust according to source */
        if (abs(slope) < 0.3)
            continue;
        /* If slope is negative, left group */
        else if (slope <= 0.0) {
            left_line_x.insert(left_line_x.end(), lines[i][0]);
            left_line_x.insert(left_line_x.end(), lines[i][2]);
            left_line_y.insert(left_line_y.end(), lines[i][1]);
            left_line_y.insert(left_line_y.end(), lines[i][3]);
        }
        else {
            right_line_x.insert(right_line_x.end(), lines[i][0]);
            right_line_x.insert(right_line_x.end(), lines[i][2]);
            right_line_y.insert(right_line_y.end(), lines[i][1]);
            right_line_y.insert(right_line_y.end(), lines[i][3]);
        }
    }

    int left_x_start, left_x_end;
    int right_x_start, right_x_end;

    left_x_start = poly1d_eval(polyfit_boost(left_line_y, left_line_x, 1), max_y);
    left_x_end   = poly1d_eval(polyfit_boost(left_line_y, left_line_x, 1), min_y);

    right_x_start = poly1d_eval(polyfit_boost(right_line_y, right_line_x, 1), max_y);
    right_x_end   = poly1d_eval(polyfit_boost(right_line_y, right_line_x, 1), min_y);

    std::vector<int> finalLines = {left_x_start,  left_x_end, 
        right_x_start, right_x_end, 
        min_y,         max_y};
    //   t = 1000*((getTickCount() - t) / getTickFrequency());
    //    std::cout << "getLinearFunction took\t" << t << " milliseconds to process the image." 
    //        << std::endl << std::endl;

    return finalLines;
}

Mat drawLines(std::vector<int> finalLines, Mat& src)
{
    line(src, Point(finalLines[0], finalLines[5]), 
            Point(finalLines[1], finalLines[4]), 
            Scalar(0, 255, 0), 7, LINE_AA);
    line(src, Point(finalLines[2], finalLines[5]), 
            Point(finalLines[3], finalLines[4]), 
            Scalar(0, 255, 0), 7, LINE_AA);

    return src;
}

Mat drawArea(std::vector<int> areaPoints, Mat& src)
{
    Point road_area[1][4];
    road_area[0][0] = Point(areaPoints[0], areaPoints[5]);
    road_area[0][1] = Point(areaPoints[1], areaPoints[4]);
    road_area[0][2] = Point(areaPoints[3], areaPoints[4]);
    road_area[0][3] = Point(areaPoints[2], areaPoints[5]);

    const Point* ppt[1] = {road_area[0]};
    int npt[] = {4};

    Mat test = Mat(src.size(), src.type(), Scalar(255, 255, 255));
    Mat dst;
    fillPoly(test,
            ppt,
            npt,
            1,
            Scalar(0 ,255 ,0),
            LINE_AA);

    addWeighted(src, 0.8, test, 0.2, 0.0, dst);

    return dst;
}

int getBrightness(Mat& img)
{
    Mat gray;
    Scalar brightness;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    brightness = mean(gray);
    int bright = sum(brightness)[0];
    //    std::cout << "brightness = " << bright << std::endl;

    return bright;
}

int processImage(std::string filename, bool render, int fov)
{
    Mat src, src_gray;
    src = imread(filename);
    if (src.empty()) {
        std::cerr << "The image " << filename << " could not be loaded." << std::endl;
        return EXIT_FAILURE;
    }
    //    std::cout << "Processing "<< filename << "..." << std::endl;

    /* Benchmark */
    double t;
    t = (double)getTickCount();

    // 1. Frame pre-processing - common to any type of source
    // Determine brightness
    int brightness = getBrightness(src);

    // a. Grayscaling
    /* We transform the image from RGB to grayscale space so it's easier to work with. */
    cvtColor(src, src_gray, COLOR_BGR2GRAY);
    Mat edged;
    /* What if brightness is very low ? */
    if (BRIGHTNESS_L >  brightness)
        ;

    /* Use adaptive mean threshold technique to detect lanes if 100 < brightnes < 150 (
     * but what is considered bright ?) */
    else if (BRIGHTNESS_L < brightness && brightness < BRIGHTNESS_A) {
        // Strong denoising
        Mat denoised;
        denoised = denoise(src_gray);
        // d. Binarization instead of Canny ?
        /* The image is binarized. The key value here is the threshold value that
         * determines if the pixel will be black or white. */

        int cond;
        Mat gray_denoised;
        cvtColor(denoised, gray_denoised, COLOR_BGR2GRAY);
        edged = adaptBinary(gray_denoised, cond);
        // imshow("Edged", edged);
        // waitKey();
    }

    /* Use canny edge technique if 150 < brightness */
    else if (BRIGHTNESS_A < brightness) {
        /* A median filter is applied to remove noise and reduce sharpness, 
         * improving the results of the edge detector. */
        // TODO - find the best values for the bilateralFilter
        Mat smoothed = Mat::zeros(src_gray.size(), src_gray.type());
        bilateralFilter(src_gray, smoothed, 11, 75, 75, BORDER_DEFAULT);
        // imshow("Smoothed", smoothed);
        // waitKey();
        // e. Edge detection 
        /* Find the median pixel value of our image in order to get the best threshold values for 
         * our canny edge detector. */
        double median = medianMat(smoothed);
        // std::cout << "The median is " << median << std::endl;

        /* -> Before getting our ROI : we need to process the whole image to make the best
         * use of the canny edge detector
         * Canny operator is used. */
        // TODO - Look for the best way to detect edges (Canny ?)
        edged = betterCanny(smoothed, median);
        imshow("Edged", edged);
        waitKey();
    }

    // b. ROI selection - common 
    /* The top half of the image is not likely to contain any road lanes.
     * The ROI will be the lower half of the image. */
    Mat ROI = getTheROI(edged, fov);
    // imshow("ROI", ROI);
    // waitKey();

    // d. Binarization (occurs with Canny detection)
    /* The ROI is binarized. The key value here is the threshold value that
     * determines if the pixel will be black or white. */

    // 2. Lane detection via Probabilistic Hough transform
    std::vector<Vec4i> lines = HoughTransform(ROI);
    // TODO - ignore once and for all

    // 2. b - Generate one line per side
    double min_y, max_y;
    min_y = src.size().height * 0.7;
    max_y = src.size().height;
    std::vector<int> finalLines = getLinearFunction(lines, min_y, max_y);

    // Draw final lines 
    Mat srcLines = drawLines(finalLines, src);

    drawArea(finalLines, src);

    if (render) {
        imshow("srcLines", src);
        waitKey();
    }

    // generateLinearFunction(houghed);
    // line(houghed, Point(349, 1080), Point(1189, 648), 255, 2, LINE_AA);
    // line(houghed, Point(1604, 1080), Point(1082, 648), 255, 2, LINE_AA);
    // imshow("lines!", houghed);
    // waitKey();
    // 3. Lane tracking

    //    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    //    std::cout << std::endl << "The program took\t\t" << t 
    //        << " milliseconds to process the image." << std::endl;

    return 0;
}

int processVideo(std::string filename, bool render, int fov)
{
    //TODO  - Handle video file error
    //      - Show average processing time per frame
    VideoCapture cap;
    cap.open(filename);

    //std::cout << "Processing " << filename << "..." << std::endl;

    /* Benchmark */
    double t;
    t = (double)getTickCount();
    Mat src, src_gray, gray_denoised, smoothed, edged, ROI, denoised;
    for (;;) {
        cap >> src;
        // Break when there's no more frames
        if (src.empty()) break;
        // 1. Frame pre-processing
        // Determine brightness

        // 1. Frame pre-processing - common to any type of source
        // Determine brightness
        int brightness = getBrightness(src);

        // a. Grayscaling
        /* We transform the image from RGB to grayscale space so it's easier to work with. */
        cvtColor(src, src_gray, COLOR_BGR2GRAY);
        /* What if brightness is very low ? */
        if (BRIGHTNESS_L > brightness) {
            ;
        }

        /* Use adaptive mean threshold technique to detect lanes if 100 < brightnes < 150 (
         * but what is considered bright ?) */
        else if (BRIGHTNESS_L < brightness && brightness < BRIGHTNESS_A) {
            // Strong denoising
            gray_denoised = denoise(src_gray);
            // d. Binarization instead of Canny ?
            /* The image is binarized. The key value here is the threshold value that
             * determines if the pixel will be black or white. */

            int cond;
            edged = adaptBinary(gray_denoised, cond);
            // imshow("Edged", edged);
            // waitKey();
        }

        /* Use canny edge technique if 150 < brightness */
        else if (BRIGHTNESS_A < brightness) {
            /* A median filter is applied to remove noise and reduce sharpness, 
             * improving the results of the edge detector. */
            // TODO - find the best values for the bilateralFilter
            smoothed = Mat::zeros(src_gray.size(), src_gray.type());
            bilateralFilter(src_gray, smoothed, 5, 75, 75, BORDER_DEFAULT);
            // imshow("Smoothed", smoothed);
            // waitKey();
            // e. Edge detection 
            /* Find the median pixel value of our image in order to get the best threshold values for 
             * our canny edge detector. */
            double median = medianMat(smoothed);
            // std::cout << "The median is " << median << std::endl;

            /* -> Before getting our ROI : we need to process the whole image to make the best
             * use of the canny edge detector
             * Canny operator is used. */
            // TODO - Look for the best way to detect edges (Canny ?)
            edged = betterCanny(smoothed, median);
        }

        // b. ROI selection - common 
        /* The top half of the image is not likely to contain any road lanes.
         * The ROI will be the lower half of the image. */ 
        ROI = getTheROI(edged, fov);
        // imshow("ROI", ROI);
        // waitKey();

        // 2. Lane detection via Probabilistic Hough transform
        std::vector<Vec4i> lines = HoughTransform(ROI);
        // TODO - ignore once and for all

        // 2. b - Generate one line per side
        double min_y, max_y;
        min_y = src.size().height * 0.7;
        max_y = src.size().height;
        std::vector<int> finalLines = getLinearFunction(lines, min_y, max_y);

        // Draw final lines 
        Mat srcLines = drawLines(finalLines, src);
        drawArea(finalLines, src);

        /* Render */
        if (render) { 
            imshow("area", src); 
            // Once we have displayed the frame, we then wait 33 ms. 
            //If a key is pressed during that
            // time we exit the read loop.
            if (waitKey(1) == 113) {
                cap.release();
                break;
            }
        }
        // 3. Lane tracking
    }

    /* Display total time */
    // t = 1000*((double)getTickCount() - t)/getTickFrequency();
    // std::cout << std::endl << "The program took\t" << t 
    //     << " milliseconds to process the video." << std::endl;

    return 0;
}







