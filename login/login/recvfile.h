#ifndef RECVFILE_H
#define RECVFILE_H

#include <QThread>
#include <QTcpSocket>
#include <QFile>

class RecvFile : public QThread
{
    Q_OBJECT
public:
    explicit RecvFile(QTcpSocket* tcp, QObject *parent = nullptr);

protected:
    // 添加从父类继承来的虚函数 run()
    void run() override;

private:
    QTcpSocket* m_tcp;

signals:
    void over();

};

#endif // RECVFILE_H
