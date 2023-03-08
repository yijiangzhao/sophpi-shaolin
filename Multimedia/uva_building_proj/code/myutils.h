#ifndef _UTILS_H_
#define _UTILS_H_
#include <ctime>
#include <string>
#include <opencv2/core.hpp>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
using namespace std;

string getTimeStampString();
string getNewViodeClipName(const char* path = "./",const char* container = ".mp4");
bool isPointInRect(cv::Point2f center_point,cv::Rect2f rect);
bool isDirExist(string dirpath);
bool create_dir(string dirpath);
#endif