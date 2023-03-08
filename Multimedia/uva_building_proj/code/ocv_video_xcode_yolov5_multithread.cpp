/**
  @file videocapture_basic.cpp
  @brief A very basic sample for using VideoCapture and VideoWriter
  @author PkLab.net
  @date Aug 24, 2016
*/
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <stdio.h>
#include <queue>

#include "yolov5.hpp"

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#include <signal.h>
#endif
#define IMAGE_MATQUEUE_NUM 50

#include "myutils.h"

#define BM_ALIGN16(_x)             (((_x)+0x0f)&~0x0f)
#define BM_ALIGN32(_x)             (((_x)+0x1f)&~0x1f)
#define BM_ALIGN64(_x)             (((_x)+0x3f)&~0x3f)


using namespace cv;
using namespace std;

typedef struct  threadArg{
    string inputUrl1;
    string inputUrl2;
    string videoSavePath;
    string codecType;
    int         frameNum;
    string outputName;
    int         yuvEnable;
    int         roiEnable;
    int         deviceId;
    string encodeParams;
    int         startWrite;
    int         fps;
    int         imageCols;
    int         imageRows;
    queue<Mat*> *imageQueue;
    queue<vector<Rect_<float>>> *detectedAreaQueue;
}THREAD_ARG;

class SliceVideoWriter {
  private:
    unsigned int frameCount;
    double fps;
    VideoWriter writer;
    int framesPer5Sec;
    int fourcc;
    unsigned int videoClipsCount;
    std::string encodeparams;
    std::string videoClipsSavePath;
  public:
    bool open(const String& videoSavePath, int fourcc, double fps,
                      Size frameSize, const String& encodeParams, bool isColor = true, int id=0){
                        writer.open(getNewViodeClipName(videoClipsSavePath.c_str()),fourcc,fps,frameSize,encodeParams,isColor);
                        this->videoClipsSavePath = videoSavePath;
                        this->fps = fps;
                        this->frameCount = 0;
                        this->fourcc = fourcc;
                        this->framesPer5Sec = fps * 5;
                        this->videoClipsCount = 0;
                        this->encodeparams = encodeParams;
                      }
    void write(InputArray image, char *data, int *len, CV_RoiInfo *roiinfo){
        writer.write(image,data,len,roiinfo);
        if (!((++frameCount) % framesPer5Sec))
        {
            cout << "slice !"<<endl;
            writer.release();
           // output_file = getNewViodeClipName();
            // writer.open(getNewViodeClipName(), fourcc, fps, image.size(), true);
            writer.open(getNewViodeClipName(videoClipsSavePath.c_str()),fourcc,fps,image.size(),encodeparams,true);
            if (!writer.isOpened())
            {
                cerr << "Could not open the output video file for write\n";
                // return -1;
            }
            videoClipsCount++;
        }
    }
    void write(InputArray image){
        writer.write(image);
        if (!((++frameCount) % framesPer5Sec))
        {
            cout << "slice !"<<endl;
            writer.release();
           // output_file = getNewViodeClipName();
            writer.open(getNewViodeClipName(videoClipsSavePath.c_str()),fourcc,fps,image.size(),encodeparams,true);
            if (!writer.isOpened())
            {
                cerr << "Could not open the output video file for write\n";
            }
            videoClipsCount++;
        }
    }
    void write(InputArray image, char *data, int *len){
        writer.write(image,data,len);
        if (!((++frameCount) % framesPer5Sec))
        {
            cout << "slice !"<<endl;
            writer.release();
           // output_file = getNewViodeClipName();
            writer.open(getNewViodeClipName(videoClipsSavePath.c_str()),fourcc,fps,image.size(),encodeparams,true);
            if (!writer.isOpened())
            {
                cerr << "Could not open the output video file for write\n";
            }
            videoClipsCount++;
        }
    }

    bool isOpened(){
      return writer.isOpened();
    }

    void release(){
      writer.release();
    }
};


//////////////////////////////////////////
typedef struct inputData{
    Mat* img;
    Mat* img2;
    vector<Rect_<float>> rects;
}inputData;
// yolo thread data
typedef struct  YOLOThreadArg{
    YoloV5      *ptrYolov5;
    int         maxBatch;
    queue<inputData> *inputImageQueue;
    queue<Mat*> *outputImageQueue;
    queue<vector<Rect_<float>>> *outputDetectedAreaQueue;
}YOLO_THREAD_ARG;
////////////////////////////////////

//std::queue<Mat> g_image_queue;
std::mutex g_video_lock;
////////////////////////////////////////
// ente:mutex for yolov5 read imgs
std::mutex yolo_lock;

int exit_flag = 0;
#ifdef __linux__
void signal_handler(int signum){
    exit_flag = 1;
}
#elif _WIN32
static BOOL CtrlHandler(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
    case CTRL_C_EVENT:
        exit_flag = 1;
        return(TRUE);
    default:
        return FALSE;
    }
}
#endif
////////////////////////////////////////
void resizeDiff(vector<Rect_<float>> rects, Mat origin, Mat &Dst){
    for (auto rect : rects){
        int x1=0, x2 = 0, y1 = 0, y2 = 0;
        if (rect.width < 50 || rect.height < 50) {
            x1 = rect.x - rect.width;
            y1 = rect.y - rect.height;
            x2 = rect.x + rect.width * 2;
            y2 = rect.y + rect.height * 2;
        }
        else {
            x1 = rect.x - rect.width/2;
            y1 = rect.y - rect.height/2;
            x2 = rect.x + rect.width * 3/2;
            y2 = rect.y + rect.height * 3/2;
        }
        if (x1 > 0 && y1 > 0 && x2 < origin.cols*4/5 && y2 < origin.rows) {
            int newWidth = x2 - x1;
            int newHeight = y2 - y1;
            cv::Mat imgROI = origin(rect);
            cv::Mat tranPart = Dst(cv::Range(y1, y2), cv::Range(x1, x2));
            cv::Mat roiResize;
            cv::resize(imgROI, roiResize, cv::Size(tranPart.cols, tranPart.rows));
            roiResize.copyTo(tranPart);
            cv::rectangle(Dst, cv::Point(x1, y1),
                cv::Point(x2, y2),
                cv::Scalar(255, 0, 0), 2, 8);
        }            
    }
}

// concat two image to dst
void concat(bm_handle_t handle, Mat *img1, Mat *img2, Mat *dst){
    bm_image bmimg1, bmimg2;
    vector<bm_image> in;
    cv::bmcv::toBMI((Mat &)(*img1), &bmimg1);
    cv::bmcv::toBMI((Mat &)(*img2), &bmimg2);
    in.push_back(bmimg1);
    in.push_back(bmimg2);
    bm_image stitch_image;
    bm_image_create(handle, 2*bmimg1.height, 
        bmimg1.width, bmimg1.image_format, bmimg1.data_type, &stitch_image);
    bm_image_alloc_dev_mem(stitch_image, BMCV_HEAP1_ID);
    std::vector<bmcv_rect_t> srt;
    std::vector<bmcv_rect_t> drt;
    bmcv_rect_t rt;
    for(int j=0;j<2;j++){
        rt.start_x = rt.start_y = 0;
        rt.crop_w  = img1->cols;
        rt.crop_h  = img1->rows;
        srt.push_back(rt);

        rt.start_x = 0;
        rt.crop_w  = img1->cols;
        rt.start_y = (j % 2) * (img1->rows);
        rt.crop_h  = img1->rows;
        drt.push_back(rt);
    }
    auto ret = bmcv_image_vpp_stitch(handle,
                                2,
                                &in[0],
                                stitch_image,
                                &drt[0]);
    assert(BM_SUCCESS == ret);
    Mat *concat_img = new Mat;
    cv::bmcv::toMAT(&stitch_image, *concat_img, true);
    *dst = concat_img->clone();
    delete concat_img;
    bm_image_destroy(bmimg1);
    bm_image_destroy(bmimg2);
    bm_image_destroy(stitch_image);
}
// ente: thread func for processing imgs
void * yoloProcessThread(void *arg){
    YOLO_THREAD_ARG *threadPara = (YOLO_THREAD_ARG *)(arg);
    YoloV5 *yolo = threadPara->ptrYolov5;
    cout<<"YOLOV5 process start"<<endl;
    int quit_times         = 0;
    Mat *toProcImage;
    Mat *toProcImage2;
    bm_handle_t handle;
    bm_dev_request(&handle, 0);
    while(1){
        // processing        
        if(!threadPara->inputImageQueue->empty()){
            std::vector<cv::Mat> images;
            std::vector<YoloV5BoxVec> boxes;
            vector<Rect_<float>> detectedRects;
            yolo_lock.lock();
            toProcImage = threadPara->inputImageQueue->front().img;
            toProcImage2 = threadPara->inputImageQueue->front().img2;
            vector<Rect_<float>> rects = threadPara->inputImageQueue->front().rects;
            images.push_back(*toProcImage );
            yolo_lock.unlock();

            if(images.size()>0){
            // ente:AI inference
            //cout<<"YOLOV5 process IMG"<<endl;
            CV_Assert(0 == yolo->Detect(images, boxes));
            for (int i = 0; i < (int) images.size(); ++i) {
                Mat *frame = new Mat;
                *frame = images[i].clone();
                yolo->draw_objects(*frame, boxes[i]);
                // for (auto bbox : boxes[i]) {
                //     yolo->drawPred(bbox.class_id, bbox.score, bbox.x, bbox.y, bbox.x + bbox.width,
                //         bbox.y + bbox.height, *frame);
                // }
                // // resizeDiff(rects, images[i], *frame);
                for (auto rect : rects) {
                    cv::rectangle(*frame, cv::Point(rect.x, rect.y),
                                cv::Point(rect.x+ rect.width, rect.y + rect.height),
                                cv::Scalar(255, 0, 0), 2, 8);
                }
                Mat *out = new Mat;
                *out = frame->clone();
                //concat(handle, toProcImage2,frame, out);

                for(auto box : boxes[i]){
                    detectedRects.push_back(box.rect);
                }

                g_video_lock.lock();
                threadPara->outputImageQueue->push(out);
                threadPara->outputDetectedAreaQueue->push(detectedRects);
                delete toProcImage;
                delete toProcImage2;
                g_video_lock.unlock();
                delete frame;
                }
            }
            yolo_lock.lock();
            threadPara->inputImageQueue->pop();
            yolo_lock.unlock();
            quit_times = 0;
        }
        else{
#ifdef __linux__
            usleep(2000);
#elif _WIN32
            Sleep(2);
#endif
            quit_times++;
        }

        if((exit_flag && threadPara->inputImageQueue->size() == 0) || quit_times >= 400){//No bitstream exits after a delay of three seconds
            break;
        }
        
    }  
#ifdef __linux__
    return (void *)0;
#elif _WIN32
    return -1;
#endif  
}
////////////////////////////////////////

#ifdef __linux__
void *videoWriteThread(void *arg){
#elif _WIN32
DWORD WINAPI videoWriteThread(void* arg){
#endif
    THREAD_ARG *threadPara = (THREAD_ARG *)(arg);
    FILE *fp_out           = NULL;
    char *out_buf          = NULL;
    int is_stream          = 0;
    int quit_times         = 0;
    VideoWriter            writer;
    SliceVideoWriter       sliceWriter;
    Mat                    image;
    string outfile         = "";
    string encodeparms     = "";
    int64_t curframe_start = 0;
    int64_t curframe_end   = 0;
    Mat *toEncImage;
    vector<Rect_<float>>    detectAreaRect;
#ifdef __linux__
    struct timeval         tv;
#endif
    if(threadPara->encodeParams.c_str())
        encodeparms = threadPara->encodeParams;

    if((strcmp(threadPara->outputName.c_str(),"NULL") != 0) && (strcmp(threadPara->outputName.c_str(),"null") != 0))
        outfile = threadPara->outputName;

    if(strstr(threadPara->outputName.c_str(),"rtmp://") || strstr(threadPara->outputName.c_str(),"rtsp://"))
        is_stream = 1;

    if(strcmp(threadPara->codecType.c_str(),"H264enc") ==0)
    {
        writer.open(outfile, VideoWriter::fourcc('a', 'v', 'c', '1'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true,
        threadPara->deviceId);

        sliceWriter.open(threadPara->videoSavePath, 
        VideoWriter::fourcc('a', 'v', 'c', '1'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true,
        threadPara->deviceId);

    }
    else if(strcmp(threadPara->codecType.c_str(),"H265enc") ==0)
    {
       writer.open(outfile, VideoWriter::fourcc('h', 'v', 'c', '1'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true,
        threadPara->deviceId);

        sliceWriter.open(threadPara->videoSavePath,
        VideoWriter::fourcc('h', 'v', 'c', '1'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true);
    }
    else if(strcmp(threadPara->codecType.c_str(),"MPEG2enc") ==0)
    {
       writer.open(outfile, VideoWriter::fourcc('M', 'P', 'G', '2'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        true,
        threadPara->deviceId);

        sliceWriter.open(threadPara->videoSavePath,
        VideoWriter::fourcc('M', 'P', '6', '2'),
        threadPara->fps,
        Size(threadPara->imageCols, threadPara->imageRows),
        encodeparms,
        true);
    }

    if(!writer.isOpened())
    {
#ifdef __linux__
        return (void *)-1;
#elif _WIN32
        return -1;
#endif
    }

    while(1){
        if(is_stream){
#ifdef __linux__
            gettimeofday(&tv, NULL);
            curframe_start= (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
#elif _WIN32
            FILETIME ft;
            GetSystemTimeAsFileTime(&ft);
            curframe_start = (int64_t)ft.dwHighDateTime << 32 | ft.dwLowDateTime;// unit is 100 us
            curframe_start = curframe_start / 10;
#endif
        }

        //if(threadPara->startWrite && !g_image_queue.empty()) {
        if(threadPara->startWrite && !threadPara->imageQueue->empty()) {
            if((strcmp(threadPara->outputName.c_str(),"NULL") == 0) || (strcmp(threadPara->outputName.c_str(),"null") == 0)){
                g_video_lock.lock();
                //writer.write(g_image_queue.front());
                toEncImage = threadPara->imageQueue->front();
                detectAreaRect = threadPara->detectedAreaQueue->front();
                g_video_lock.unlock();
                writer.write(*toEncImage);
                sliceWriter.write(*toEncImage);
            }else{
                if(fp_out == NULL){
                        fp_out = fopen("pkt.dump","wb+");
                }
                if(out_buf == NULL){
                        out_buf = (char*)malloc(threadPara->imageCols * threadPara->imageRows * 4);
                }
                int out_buf_len = 0;
                //writer.write(g_image_queue.front(),out_buf,&out_buf_len);
                g_video_lock.lock();
                toEncImage = threadPara->imageQueue->front();
                detectAreaRect = threadPara->detectedAreaQueue->front();
                g_video_lock.unlock();
                if (threadPara->roiEnable == 1) {
                    static unsigned int roi_frame_nums = 0;
                    roi_frame_nums++;
                    CV_RoiInfo  roiinfo;
                    if (strcmp(threadPara->codecType.c_str(),"H264enc") ==0) {
                        int nums = (BM_ALIGN16(threadPara->imageRows) >> 4) * (BM_ALIGN16(threadPara->imageCols) >> 4);
                        roiinfo.numbers = nums;
                        roiinfo.customRoiMapEnable = 1;
                        roiinfo.field = (cv::RoiField*)malloc(sizeof(cv::RoiField)*nums);
                        for (int i = 0;i <(BM_ALIGN16(threadPara->imageRows) >> 4);i++) {
                            for (int j=0;j < (BM_ALIGN16(threadPara->imageCols) >> 4);j++) {
                                int pos = i*(BM_ALIGN16(threadPara->imageCols) >> 4) + j;
                                for(auto rect : detectAreaRect)
                                {
                                    if(isPointInRect(cv::Point2f(j*16+8,i*16+8),rect))
                                    {
                                        roiinfo.field[pos].H264.mb_qp = 10;
                                        // 绘出roi区域
                                        // cv::rectangle(*toEncImage, cv::Point(j*16, i*16),
                                        //     cv::Point(j*16 + 16, i*16+16),
                                        //     cv::Scalar(255, 255, 0), 5, 8);
                                        break;
                                    }
                                    else
                                    {
                                        roiinfo.field[pos].H264.mb_qp = 40;
                                    }
                                }
                            }
                        }
                    } else if (strcmp(threadPara->codecType.c_str(),"H265enc") ==0) {
                        int nums = (BM_ALIGN64(threadPara->imageRows) >> 6) * (BM_ALIGN64(threadPara->imageCols) >> 6);
                        roiinfo.numbers = nums;
                        roiinfo.field = (cv::RoiField*)malloc(sizeof(cv::RoiField)*nums);
                        roiinfo.customRoiMapEnable    = 1;
                        roiinfo.customModeMapEnable   = 0;
                        roiinfo.customLambdaMapEnable = 0;
                        roiinfo.customCoefDropEnable  = 0;

                        for (int i = 0;i <(BM_ALIGN64(threadPara->imageRows) >> 6);i++) {
                            for (int j=0;j < (BM_ALIGN64(threadPara->imageCols) >> 6);j++) {
                                int pos = i*(BM_ALIGN64(threadPara->imageCols) >> 6) + j;
                                if ((i >= (BM_ALIGN64(threadPara->imageRows) >> 6)/2) && (j >= (BM_ALIGN64(threadPara->imageCols) >> 6)/2)) {
                                    roiinfo.field[pos].HEVC.sub_ctu_qp_0 = 10;
                                    roiinfo.field[pos].HEVC.sub_ctu_qp_1 = 10;
                                    roiinfo.field[pos].HEVC.sub_ctu_qp_2 = 10;
                                    roiinfo.field[pos].HEVC.sub_ctu_qp_3 = 10;
                                } else {
                                    roiinfo.field[pos].HEVC.sub_ctu_qp_0 = 40;
                                    roiinfo.field[pos].HEVC.sub_ctu_qp_1 = 40;
                                    roiinfo.field[pos].HEVC.sub_ctu_qp_2 = 40;
                                    roiinfo.field[pos].HEVC.sub_ctu_qp_3 = 40;

                                }
                                roiinfo.field[pos].HEVC.ctu_force_mode = 0;
                                roiinfo.field[pos].HEVC.ctu_coeff_drop = 0;
                                roiinfo.field[pos].HEVC.lambda_sad_0 = 0;
                                roiinfo.field[pos].HEVC.lambda_sad_1 = 0;
                                roiinfo.field[pos].HEVC.lambda_sad_2 = 0;
                                roiinfo.field[pos].HEVC.lambda_sad_3 = 0;
                            }
                        }
                    }
                    writer.write(*toEncImage,out_buf,&out_buf_len, &roiinfo);
                    sliceWriter.write(*toEncImage,out_buf,&out_buf_len, &roiinfo);
                }
                else {
                    writer.write(*toEncImage,out_buf,&out_buf_len);
                    sliceWriter.write(*toEncImage,out_buf,&out_buf_len);
                }
                if(out_buf_len > 0){
                    fwrite(out_buf,1,out_buf_len,fp_out);
                }
            }

            g_video_lock.lock();
            //g_image_queue.pop();
            threadPara->imageQueue->pop();
            threadPara->detectedAreaQueue->pop();
            delete toEncImage;
            g_video_lock.unlock();
            quit_times = 0;
        }else{
#ifdef __linux__
            usleep(2000);
#elif _WIN32
            Sleep(2);
#endif
            quit_times++;
        }
        //only Push video stream
        if(is_stream){
#ifdef __linux__
            gettimeofday(&tv, NULL);
            curframe_end= (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
#elif _WIN32
            FILETIME ft;
            GetSystemTimeAsFileTime(&ft);
            curframe_end = (int64_t)ft.dwHighDateTime << 32 | ft.dwLowDateTime;
            curframe_end = curframe_end / 10;
#endif
            if(curframe_end - curframe_start > 1000000 / threadPara->fps)
                continue;
#ifdef __linux__
            usleep((1000000 / threadPara->fps) - (curframe_end - curframe_start));
#elif _WIN32
            Sleep((1000 / threadPara->fps) - (curframe_end - curframe_start)/1000 - 1);
#endif
        }
        //if((exit_flag && g_image_queue.size() == 0) || quit_times >= 1000){//No bitstream exits after a delay of three seconds
        if((exit_flag && threadPara->imageQueue->size() == 0) || quit_times >= 400){//No bitstream exits after a delay of three seconds
            break;
        }
    }
    writer.release();
    if(fp_out != NULL){
        fclose(fp_out);
        fp_out = NULL;
    }

    if(out_buf != NULL){
        free(out_buf);
        out_buf = NULL;
    }
#ifdef __linux__
    return (void *)0;
#elif _WIN32
    return -1;
#endif
}

vector<Rect_<float>> diff(Mat frame1, Mat frame2){
    vector<Rect_<float>> vrects;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    int min_area = 50;
    Mat diff;
    cv::absdiff(frame1, frame2, diff);
    cv::threshold(diff, diff, 40, 255, cv::THRESH_BINARY);
    cv::morphologyEx(diff, diff, cv::MORPH_CLOSE, element);
    std::vector<std::vector<cv::Point> > contours;  // 创建轮廓容器
    std::vector<cv::Vec4i> 	hierarchy;
    cv::findContours(diff, contours, hierarchy,
        cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());
    
    if (!contours.empty() && !hierarchy.empty())
    {
        std::vector<std::vector<cv::Point> >::const_iterator itc = contours.begin();
        // 遍历所有轮廓
        while (itc != contours.end())
        {
            // 定位当前轮廓所在位置
            cv::Rect rect = cv::boundingRect(cv::Mat(*itc));
            // contourArea函数计算连通区面积
            double area = contourArea(*itc);
            // 若面积小于设置的阈值
            if (area > min_area)
                vrects.push_back(rect);
            itc++;
        }
    }
    return vrects;
}


int main(int argc, char* argv[])
{
    const char *keys="{bmodel | ./yolov5s_building_seg.bmodel | bmodel file path}"
    "{code_type | H264enc | H264enc is h264; H265enc is h265.}"
    "{yuv_enable | 1 | 0 decode output bgr; 1 decode output yuv420.}"
    "{roi_enable | 0 | 0 disable roi encoder; 1 enable roi encoder.if roi_enable is 1, you should set null/Null in outputname and set roi_eanble=1 in encodeparams.}"
    "{outputName | results.mp4 | output path, null or NULL output pkt.dump.}"
    "{encodeparams | bitrate=800 | gop=30:bitrate=800:gop_preset=2:mb_rc=1:delta_qp=3:min_qp=20:max_qp=40:roi_enable=1:push_stream=rtmp/rtsp.}"
    "{tpuid | 0 | TPU device id}"
    "{conf | 0.5 | confidence threshold for filter boxes}"
    "{obj | 0.5 | object score threshold for filter boxes}"
    "{iou | 0.5 | iou threshold for nms}"
    "{help | 0 | Print help information.}"
    "{frameNum | 1000 | number of frames in video to process, 0 means processing all frames}"
    "{input1 | uav_720p_1.mp4 | input stream file1 path}"
    "{input2 | uav_720p_1.mp4 | input stream file2 path}"
    "{videoSavePath | ./ | the path where the video clips is stored }"
    "{classnames | uva.names | class names' file path}";

    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.get<bool>("help")) {
        parser.printMessage();
        return 0;
    }
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap1;
    VideoCapture cap2;

#ifdef WIN32
    LARGE_INTEGER liPerfFreq={0};
    QueryPerformanceFrequency(&liPerfFreq);
    LARGE_INTEGER tv1 = {0};
    LARGE_INTEGER tv2 = {0};
    HANDLE threadId;
#else
    struct timeval tv1, tv2;
    pthread_t threadId;
#endif

    THREAD_ARG *threadPara = (THREAD_ARG *)malloc(sizeof(THREAD_ARG));
    memset(threadPara,0,sizeof(THREAD_ARG));

#ifdef __linux__
    signal(SIGINT, signal_handler);
#elif _WIN32
    SetConsoleCtrlHandler((PHANDLER_ROUTINE)CtrlHandler, TRUE);
#endif
    if(!threadPara){
        return -1;
    }
    threadPara->imageQueue   = new queue<Mat*>;
    threadPara->detectedAreaQueue = new queue<vector<Rect_<float>>>;
    threadPara->outputName = parser.get<std::string>("outputName");
    threadPara->codecType = parser.get<std::string>("code_type");
    threadPara->frameNum   = parser.get<int>("frameNum");

    cout<<"threadPara->frameNum: "<<threadPara->outputName<<endl;
    cout<<"threadPara->codecType: "<<threadPara->codecType<<endl;
    cout<<"threadPara->frameNum: "<<threadPara->frameNum<<endl;

    if ( strcmp(threadPara->codecType.c_str(),"H264enc") ==0
      || strcmp(threadPara->codecType.c_str(),"H265enc") == 0
      || strcmp(threadPara->codecType.c_str(),"MPEG2enc") == 0)
    {
        threadPara->startWrite = 1;
    } else {
        if(threadPara->imageQueue){
            delete threadPara->imageQueue;
            threadPara->imageQueue = NULL;
        }
        if(threadPara){
            free(threadPara);
            threadPara = NULL;
        }
        return 0;
    }
#ifdef USING_SOC
    threadPara->encodeParams = parser.get<std::string>("encodeparams");
#else
    threadPara->deviceId = parser.get<int>("tpuid");
    threadPara->encodeParams = parser.get<std::string>("encodeparams");
#endif
    cout<<"threadPara->encodeParams: "<<threadPara->encodeParams<<endl;
    
    threadPara->yuvEnable = parser.get<int>("yuv_enable");
    if ((threadPara->yuvEnable != 0) && (threadPara->yuvEnable != 1)) {
        cout << "yuv_enable param err." << endl;
        return -1;
    }
    threadPara->roiEnable = parser.get<int>("roi_enable");
    if ((threadPara->roiEnable != 0) && (threadPara->roiEnable != 1)) {
        cout << "roi_enable param err." << endl;
        return -1;
    }

    cout<<"threadPara->yuvEnable:"<<threadPara->yuvEnable<<endl;
    cout<<"threadPara->roiEnable:"<<threadPara->roiEnable<<endl;

    threadPara->videoSavePath = parser.get<std::string>("videoSavePath");
    cout<<"threadPara->videoSavePath:"<<threadPara->videoSavePath<<endl;
    if(!isDirExist(threadPara->videoSavePath)){
        if(!create_dir(threadPara->videoSavePath))
        {
            cerr << "ERROR! invalid video clips save dir path\n";
            return -1;
        }
    }
    // open the default camera using default API
    threadPara->inputUrl1 = parser.get<std::string>("input1");
    threadPara->inputUrl2 = parser.get<std::string>("input2");
    cap1.open(threadPara->inputUrl1, CAP_FFMPEG, threadPara->deviceId);
    cap2.open(threadPara->inputUrl2, CAP_FFMPEG, threadPara->deviceId);
    if (!cap1.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    
/////////////////////////////////////////////////////////////////////////////////////////
    //YOLOV5
    // profiling
    TimeStamp ts;
    std::string bmodel_file = parser.get<std::string>("bmodel");
    std::string coco_names = parser.get<std::string>("classnames");
    int dev_id = parser.get<int>("tpuid");;
    cout<<"read yolov5 model from "<<bmodel_file<<endl;
    cout<<"read coco file from "<<coco_names<<endl;
    BMNNHandlePtr handle = std::make_shared<BMNNHandle>(dev_id);
    // Load bmodel
    std::shared_ptr<BMNNContext> bm_ctx = std::make_shared<BMNNContext>(handle, bmodel_file.c_str());
    YoloV5 yolo(bm_ctx);
    yolo.enableProfile(&ts);
    CV_Assert(0 == yolo.Init(parser.get<float>("conf"),
                            parser.get<float>("obj"),
                            parser.get<float>("iou"),
                            coco_names));
////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////
// ente: yolo data
    YOLO_THREAD_ARG *yoloThreadPara = (YOLO_THREAD_ARG *)malloc(sizeof(YOLO_THREAD_ARG));
    memset(yoloThreadPara,0,sizeof(YOLO_THREAD_ARG));
    if(!yoloThreadPara){
        return -1;
    }
    yoloThreadPara->inputImageQueue   = new queue<inputData>;
    yoloThreadPara->outputImageQueue   = threadPara->imageQueue;
    yoloThreadPara->outputDetectedAreaQueue      = threadPara->detectedAreaQueue;
    yoloThreadPara->maxBatch          = yolo.return_maxBatch();
    yoloThreadPara->ptrYolov5         = &yolo;
    //cout<<"max batch is "<< yoloThreadPara->maxBatch <<endl;
///////////////////////////////////////////////////
// ente:thread
    pthread_t threadYolo;
//////////////////////////////////////////////////
    // Set Resamper
    cap1.set(CAP_PROP_OUTPUT_SRC, 1.0);
    cap2.set(CAP_PROP_OUTPUT_SRC, 1.0);
    double out_sar = cap1.get(CAP_PROP_OUTPUT_SRC);
    cout << "CAP_PROP_OUTPUT_SAR: " << out_sar << endl;

    if(threadPara->yuvEnable == 1){
        cap1.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
        cap2.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
    }

    // Set scalar size
    //int height = (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    //int width  = (int) cap.get(cv::CAP_PROP_FRAME_WIDTH);
    cout << "orig CAP_PROP_FRAME_HEIGHT: " << (int) cap1.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "orig CAP_PROP_FRAME_WIDTH: " << (int) cap1.get(CAP_PROP_FRAME_WIDTH) << endl;

    if(threadPara->startWrite)
    {
        Mat img_tmp1;
        cap1.read(img_tmp1);
        Mat img_tmp2;
        cap2.read(img_tmp2);
        threadPara->fps = 10; //cap1.get(CAP_PROP_FPS);
        threadPara->imageCols = img_tmp1.cols;
        threadPara->imageRows = img_tmp1.rows;
        
#ifdef WIN32
        threadId = CreateThread(NULL, 0, videoWriteThread, threadPara, 0, NULL);
#else
        pthread_create(&threadId, NULL, videoWriteThread, threadPara);
        pthread_create(&threadYolo, NULL, yoloProcessThread, yoloThreadPara);
#endif
    }

    //--- GRAB AND WRITE LOOP
#ifdef WIN32
    QueryPerformanceCounter(&tv1);
#else
    gettimeofday(&tv1, NULL);
#endif
    for (int i=0; i < threadPara->frameNum; i++)
    {
        if(exit_flag){
            break;
        }
        //Mat image;
        Mat *image = new Mat;
        Mat *image2 = new Mat;
        cap1.read(*image);
        cap2.read(*image2);
        // check if we succeeded
        //if (image.empty()) {
        if (image->empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            if ((int)cap1.get(CAP_PROP_STATUS) == 2) {     // eof
                cout << "file ends!" << endl;
                cap1.release();
                cap2.release();
                cap1.open(threadPara->inputUrl1, CAP_FFMPEG, threadPara->deviceId);
                cap2.open(threadPara->inputUrl2, CAP_FFMPEG, threadPara->deviceId);
                if(threadPara->yuvEnable == 1){
                    cap1.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
                    cap2.set(cv::CAP_PROP_OUTPUT_YUV, PROP_TRUE);
                }
                cout << "loop again " << endl;
            }
            continue;
        }
/////////////////////////////////////////////////////////////////////////////////////////
        vector<Rect_<float>> rects =  diff(*image, *image2);
        inputData tmp;
        tmp.img = image;
        tmp.img2 = image2;
        tmp.rects = rects;
        
        yolo_lock.lock();
        yoloThreadPara->inputImageQueue->push(tmp);
        yolo_lock.unlock();    
        //delete image2;
/////////////////////////////////////////////////////////////////////////////////////////
        if(threadPara->startWrite){
            //while(g_image_queue.size() >= IMAGE_MATQUEUE_NUM){
            while(yoloThreadPara->inputImageQueue->size() >= IMAGE_MATQUEUE_NUM){
#ifdef __linux__
                usleep(2000);
#elif _WIN32
                Sleep(2);
#endif
                if(exit_flag){
                    break;
                }
            }
        }
        if ((i+1) % 300 == 0)
        {
            unsigned int time;
#ifdef WIN32
            QueryPerformanceCounter(&tv2);
            time = ( ((tv2.QuadPart - tv1.QuadPart) * 1000)/liPerfFreq.QuadPart);
#else
            gettimeofday(&tv2, NULL);
            time = (tv2.tv_sec - tv1.tv_sec)*1000 + (tv2.tv_usec - tv1.tv_usec)/1000;
#endif
            printf("current process is %f fps!\n", (i * 1000.0) / (float)time);
        }
    }

#ifdef WIN32
    WaitForSingleObject(threadId, INFINITE);
#else
    pthread_join(threadYolo, NULL);
    pthread_join(threadId, NULL);   
#endif
    cap1.release();
    cap2.release();
///////////////////////////////////////////////////
    if(threadPara->imageQueue){
        delete threadPara->imageQueue;
        threadPara->imageQueue = NULL;
    }
    if(threadPara->detectedAreaQueue){
        delete threadPara->detectedAreaQueue;
        threadPara->detectedAreaQueue = NULL;
    }
    if(threadPara){
        free(threadPara);
        threadPara = NULL;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
/////////////////////////////////////////////
time_stamp_t base_time = time_point_cast<microseconds>(steady_clock::now());
  ts.calbr_basetime(base_time);
  ts.build_timeline("YoloV5");
  ts.show_summary("YoloV5 Demo");
  ts.clear();
//////////////////////////////////////////////////////
// ente:release
std::cout<<"release yolo"<<std::endl;
    if(yoloThreadPara->inputImageQueue){
        delete yoloThreadPara->inputImageQueue;
        yoloThreadPara->inputImageQueue = NULL;
    }
    if(yoloThreadPara){
        free(yoloThreadPara);
        yoloThreadPara = NULL;
    }
///////////////////////////////////////////////////
    return 0;
}
