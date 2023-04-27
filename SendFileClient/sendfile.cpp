//准备要在子线程中完成的工作， 连接服务器和发送文件给服务器
#include "sendfile.h"

#include <QFile>
#include <QFileInfo>
#include <QHostAddress>
#include <QDebug>
#include <QThread>

SendFile::SendFile(QObject *parent) : QObject(parent)
{

}
// 当前是在子线程
void SendFile::connectServer(unsigned short port, QString ip)
{
    qDebug() << "连接服务器线程: " << QThread::currentThread();
    m_tcp = new QTcpSocket;
    m_tcp->connectToHost(QHostAddress(ip), port);

    // 若连接成功， 要通知主线程， 主线程弹出提示框 | 这里需要添加自定义的信号进行通知
    connect(m_tcp, &QTcpSocket::connected, this, &SendFile::connectOK);

    // 什么时候断开连接
    connect(m_tcp, &QTcpSocket::disconnected, this, [=](){
        // 关闭套接字
        m_tcp->close();
        // 释放资源
        m_tcp->deleteLater();
        // 发送信号给主线程， 告诉主线程， 服务器已经和客户端断开了连接
        emit gameover();
    });
}

void SendFile::sendFile(QString path)
{
    qDebug() << "发送文件线程: " << QThread::currentThread();
    qDebug() << "open file name" << path;
    QFile file(path);
    QFileInfo info(path);
    if (!file.open(QFile::ReadOnly))
    {
        qDebug() << "open file error!";
        return ;
    }
    int fileSize = info.size();
    qDebug() << "fileSize: " << fileSize;
    // 要把当前读取文件的进度， 发送给主窗口， 主线程根据当前的文件进度去更新进度条
    // 服务器那边可以根据文件大小的总和已经达到去判断文件是否读取结束
    // 若没有读完就一直读
    while(!file.atEnd())
    {
        static int num = 0;
        if(num == 0)
        {
            m_tcp->write((char*)&fileSize, 4);
        }
        // 读取一行就发送一行
        QByteArray line = file.readLine();
        num += line.size();
        // 计算出百分比， 发送给主线程
        int percent = (num * 100 / fileSize);
        emit curPercent(percent);
        m_tcp->write(line);
    }
}

