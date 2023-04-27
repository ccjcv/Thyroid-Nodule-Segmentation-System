#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

signals:
    void startConnect(unsigned short, QString);
    // 发送文件的信号
    void sendFile(QString path);

private slots:
    void on_connectServer_clicked();

    void on_selFile_clicked();

    void on_sendFile_clicked();

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
