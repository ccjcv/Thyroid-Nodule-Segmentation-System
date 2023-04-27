#include "recvfile.h"
#include <QFile>
#include <QDebug>

RecvFile::RecvFile(QTcpSocket* tcp, QObject *parent) : QThread(parent)
{
    m_tcp = tcp;
}

// （这里只是一个注册， 不是执行）connect 中的槽函数不知道是什么时候执行的，
//而run方法一旦执行， 直接就会走完， 当run方法执行完毕， 子线程的处理流程也就结束了
// 所以要保证当前的子线程不退出， 要一直检测事件
void RecvFile::run()
{
    qDebug() << "服务器子线程: " << QThread::currentThread();

    QFile* file = new QFile("recv.png");
    file->open(QFile::WriteOnly);

    // 接收数据
    connect(m_tcp, &QTcpSocket::readyRead, this, [=]()
    {
        static int count = 0;
        static int total = 0;
        if(count == 0)
        {
            m_tcp->read((char*)&total, 4);
        }
        // 读出剩余的数据
        QByteArray all = m_tcp->readAll();
        count += all.size();
        file->write(all);

        // 判断数据是否接收完毕了
        if(count == total)
        {
            m_tcp->close();
            m_tcp->deleteLater();
            file->close();
            file->deleteLater();
            emit over();
        }
    });

    // 进入事件循环
    //不代表子线程退出,只是到了后台,当子线程中有对应的事件触发了,
    //那事件的对应的处理功能还是在子线程中处理
    exec();
}
