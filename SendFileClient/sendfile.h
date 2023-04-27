#ifndef SENDFILE_H
#define SENDFILE_H

#include <QObject>
#include <QTcpSocket>
#include <QImageReader>
#include <QFile>

class SendFile : public QObject
{
    Q_OBJECT
public:
    explicit SendFile(QObject *parent = nullptr);


    // 连接服务器
    void connectServer(unsigned short port, QString ip);

    // 发送文件
    void sendFile(QString path);

signals:
    void connectOK();
    void gameover();
    void curPercent(int num);

private:
    QTcpSocket* m_tcp;
};

#endif // SENDFILE_H
