#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QThread>
#include "sendfile.h"
#include <QFileDialog>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString filePath;
    filePath = ":/style2.qss";

    /*皮肤设置*/
    QFile file(filePath);/*QSS文件所在的路径*/
    file.open(QFile::ReadOnly);
    QTextStream filetext(&file);
    QString stylesheet = filetext.readAll();
    this->setStyleSheet(stylesheet);
    file.close();

    qDebug() << "主线程: " << QThread::currentThread();
    ui->ip->setText("127.0.0.1");
    ui->port->setText("8989");
    ui->progressBar->setRange(0, 100);
    ui->progressBar->setValue(0);

    // 创建线程对象
    QThread* t = new QThread;
    // 创建任务对象
    SendFile* worker = new SendFile;
    //创建出来的工作的类对象移动到创建的子线程对象中
    worker->moveToThread(t);

    //connect第一个参数发送者，第二个信号，第三个接收者，第四个接收者的响应函数。

    // 工作对象接收信号
    connect(this, &MainWindow::sendFile, worker, &SendFile::sendFile);

    // 若当前的窗口对象， 发射出一个startConnect信号，
    //worker对象去接收这个信号， 调用类中的任务函数  | 信号和槽， 对应的参数要一致
    connect(this, &MainWindow::startConnect, worker, &SendFile::connectServer);
    // 处理主线程发送的信号
    connect(worker, &SendFile::connectOK, this, [=](){
         QMessageBox::information(this, "连接服务器", "已经成功连接了服务器, 恭喜!");
    });
    connect(worker, &SendFile::gameover, this, [=](){
         // 资源释放
        t->quit();
        t->wait();
        worker->deleteLater();
        t->deleteLater();
    });
    // 更新进度条的显示数据
    connect(worker, &SendFile::curPercent, ui->progressBar, &QProgressBar::setValue);

    // 启动子线程
    t->start();
}

MainWindow::~MainWindow()
{
    delete ui;
}

// 连接服务器
void MainWindow::on_connectServer_clicked()
{
    QString ip = ui->ip->text();// 获取IP值
    unsigned short port = ui->port->text().toUShort();// 获取端口值， 并转换成无符号整型
    // 发射出信号后， 让某个对应的任务函数开始执行
    emit startConnect(port, ip);
}
// 选择文件
void MainWindow::on_selFile_clicked()
{
    // 弹出选择文件对话框
    // 获得某个磁盘文件对应的绝对路径
    QString path = QFileDialog::getOpenFileName();
    if(path.isEmpty())
    {
        QMessageBox::warning(this, "打开文件", "选择的文件路径不能为空!");
        return;
    }
    // 将得到的路径设置到框中
    ui->filePath->setText(path);
}
 // 发送文件在子线程中执行， 所以主线程这里只需要发送信号给子线程中的工作对象让其完成发送文件的操作
void MainWindow::on_sendFile_clicked()
{
    emit sendFile(ui->filePath->text());
}
