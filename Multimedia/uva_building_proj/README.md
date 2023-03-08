# uva_building_proj
uva_building_proj是无人机航拍建筑物检测和对比项目。

项目由以下文件夹构成。其中，bin包含可执行文件、执行脚本、深度学习模型等；

results中包含代码运行的结果，source包含基于pytorch训练的模型、bmodel模型、nginx包，video文件夹中是测试视频，docs包含实验的指导文档。

├─bin

├─code

├─docs

├─results

├─sources

└─video

## 运行过程
从http://disk-sophgo-vip.quickconnect.cn/sharing/3SekDgy2b
使用python命令通过分享文件的url下载
命令如下：
# 安装pip包
sudo pip3 install dfn
# 下载文件
python3 -m dfn --url http://disk-sophgo-vip.quickconnect.cn/sharing/3SekDgy2b

下载完成后包括source文件和video;

并将source中的bmodel的模型复制到bin中。
拷贝bin至SE5，修改run_ocv.sh，将outputName改为自己的rtmp服务器地址。执行`bash run_ocv.sh`，在pc上用vlc查看接收的视频流。

## pc端rtmp服务器配置
### 1.安装
  1. 下载nginx：

      下载[nginx](http://nginx-win.ecsds.eu/download/)，注意，一定要选择`nginx 1.7.11.3 Gryphon.zip`这个版本，或者点[这里](http://nginx-win.ecsds.eu/download/nginx 1.7.11.3 Gryphon.zip)直接下载，据说只有这个版本的nginx在编译时是加入了[rtmp](https://so.csdn.net/so/search?q=rtmp&spm=1001.2101.3001.7020)模块的，其他版本的都没有，包括[nginx官方](http://nginx.org/)下载的也是没有包含rtmp模块的。
       也可在source中找到已经包含rtmp模块的压缩包
2. 解压

### 2.配置文件

  打开nginx/conf目录，新建一个文件：nginx.conf，然后在这个文件中输入如下内容

```text
worker_processes  1;

events {
    worker_connections  1024;
}

rtmp {
    server {
        listen 1935;
        chunk_size 4000;
        application live {
             live on;
             allow publish 127.0.0.1;
             allow play all;
        }
    }
}

```

### 3.启动，停止

  启动：在nginx目录打开cmd，输入`nginx`或`nginx -c conf/nginx.conf`来启动nginx

  ![](https://secure2.wostatic.cn/static/Bt55jqSbwNi3Qk5aaMd3J/image.png?auth_key=1675317645-pmW1qKMm27tXPHAfEwCtDZ-0-084d16138ccf178f339395e142153042)

  停止：在nginx目录打开cmd，输入`nginx -s stop`

### 4.推流

  推流地址为：`rtmp://ip:1935/live/test`，拉流地址也是这个。