#ifndef ZHU_H
#define ZHU_H

#include <QWidget>
#include <QtNetwork>
extern float out_1;
extern float out_2;
extern float out_3;
extern QString path;
namespace Ui {
class zhu;
}

class zhu : public QWidget
{
    Q_OBJECT

public:
    explicit zhu(QWidget *parent = nullptr);
    ~zhu();
    QTcpServer *tcpServer;
    QTcpSocket *currentClient;
    qint64 totalBytes;  //存放总大小信息
    qint64 bytesReceived;  //已收到数据的大小
    qint64 fileNameSize;  //文件名的大小信息
    QString fileName;   //存放文件名
    QFile *localFile;   //本地文件
    QByteArray inBlock;   //接收数据缓冲区
    qint64 bytesWritten;  //已经发送数据大小
    qint64 bytesToWrite;   //剩余数据大小
    qint64 loadSize;   //每次发送数据的大小
    QByteArray outBlock;  //数据缓冲区
//    float out_3;
public slots:
    //original NewConnection
    //void NewConnection();
    //original recMessage
    //void recMessage();
//    void sendMessage();
//    void continueSend(qint64 numBytes);

    //original disconnect
    //void disconnect();
signals:
    void back();
private slots:
    void on_setListen_clicked();
//    void on_pushButton_4_clicked();


private:
    Ui::zhu *ui;
    QTcpServer* m_s;

};

#endif // ZHU_H
