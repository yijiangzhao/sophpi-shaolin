#include "myutils.h"
#include <sstream>
#include <iostream>
using namespace std;
/// @brief 获取当前时间戳的字符串格式
/// @brief 用以切段视频命名，其格式为YearMonDay_HourMinSec
/// @return 包含时间戳信息的字符串
string getTimeStampString()
{
    string timeStamp;
    time_t now = time(NULL);
    tm *tm_t = localtime(&now);
    stringstream ss;
    ss << tm_t->tm_year + 1900;
    if (tm_t->tm_mon + 1 < 10)
        ss << "0" << tm_t->tm_mon + 1;
    else
        ss << tm_t->tm_mon + 1;

    if (tm_t->tm_mday < 10)
        ss << "0" << tm_t->tm_mday;
    else
        ss << tm_t->tm_mday;

    ss << "_";

    if (tm_t->tm_hour < 10)
        ss << "0" << tm_t->tm_hour;
    else
        ss << tm_t->tm_hour;

    if (tm_t->tm_min < 10)
        ss << "0" << tm_t->tm_min;
    else
        ss << tm_t->tm_min;

    if (tm_t->tm_sec < 10)
        ss << "0" << tm_t->tm_sec;
    else
        ss << tm_t->tm_sec;

    timeStamp = ss.str();
    return timeStamp;
}

/// @brief 获取新的视频分段的存放路径
/// @param path 想要存放的路径，默认是当前目录
/// @param container 想要存放的容器格式，默认是mp4格式
/// @return 一个存放路径的文件名
string getNewViodeClipName(const char *path, const char *container)
{
    stringstream ss;
    string newPath;
    ss << path << "/" << getTimeStampString() << container;
    newPath = ss.str();
    return newPath;
}

/// @brief 判断点是否在矩形内部（不包括边界）
/// @param center_point 要比对的点坐标
/// @param rect  要比对的矩形
/// @return 在矩形内就返回true否则返回false
bool isPointInRect(cv::Point2f center_point, cv::Rect2f rect)
{
    if (center_point.x < rect.x)
    {
        return false;
    }
    else if (center_point.x > (rect.x + rect.width))
    {
        return false;
    }
    else if (center_point.y < rect.y)
    {
        return false;
    }
    else if (center_point.y > (rect.y + rect.height))
    {
        return false;
    }
    else
    {
        return true;
    }
}

/// @brief 在当前目录下创建一个新文件夹
/// @param dirpath 目录名称
/// @return
bool create_dir(string dirpath)
{
    if (mkdir(dirpath.c_str(), 0755) == -1)
    {
        return false;
    }
    else
    {
        return true;
    }
}

/// @brief 判断目标路径文件夹是否存在
/// @param dirpath
/// @return 存在返回true 否则返回false
bool isDirExist(string dirpath)
{
    DIR *dir;
    if (access(dirpath.c_str(), NULL) != 0)
    {
        return false;
    }
    else
    {
        return true;
    }
}
