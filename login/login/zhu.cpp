#include "zhu.h"
#include "ui_zhu.h"
#include <QPushButton>
#include <QFile>
#include <QTextStream>
#include "Python.h"
#include <QFileDialog>
#include <QDebug>
#include <QMessageBox>
#include <QImageReader>
#include "recvfile.h"


zhu::zhu(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::zhu)
{
    ui->setupUi(this);

    QString filePath;
    filePath = ":/res/qss/style-3.qss";

    /*皮肤设置*/
    QFile file(filePath);/*QSS文件所在的路径*/
    file.open(QFile::ReadOnly);
    QTextStream filetext(&file);
    QString stylesheet = filetext.readAll();
    this->setStyleSheet(stylesheet);
    file.close();

    connect(ui->btn_huizhu,&QPushButton::clicked,[=](){
        emit this->back();
    });

    connect(ui->pushButton,&QPushButton::clicked,[=](){
        path = QFileDialog::getOpenFileName(this,"open file", "/home/caichengjie/QTproject/1.1/build-1_1-Desktop_Qt_5_14_2_GCC_64bit-Debug/data/tn3k_dian_2/test-image");
        ui->lineEdit->setText(path);
        ui->lbl_Image1->setPixmap(QPixmap(path));
        ui->lbl_Image1->setScaledContents(true);
    });
//    ui->lbl_test->setStyleSheet("color:red;");
//    QFont font_1("Times New Roman", 15, 75);
//    ui->lbl_test->setFont(font_1);

//    ui->lbl_title->setStyleSheet("coclor:black;");
//    QFont font_2("Times New Roman", 20, 75);
//    ui->lbl_title->setFont(font_2);

    //Unet predict
    connect(ui->pushButton_2,&QPushButton::clicked,[=](){
//        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
//        Py_SetPath(L"/home/caichengjie/anaconda3/envs/mctrans:"
//                   "/home/caichengjie/anaconda3/envs/mctrans/lib/python3.7/site-packages");
        Py_Initialize();
//        import_array();
        if ( !Py_IsInitialized() )
        {
        return -1;
        }
        PyRun_SimpleString("import sys");//设置py脚本的路径
        PyRun_SimpleString("sys.path.append('./')");//当前路径

        PyObject* pModule_detect = PyImport_ImportModule("eval_3");  // 这里的temp就是创建的python文件
        if (!pModule_detect) {
                qDebug()<< "Cant open python_detect file!\n" << endl;
                return -1;
        }
        PyObject* pFundetect = PyObject_GetAttrString(pModule_detect,"detect_img");  // 这里的hellow就是python文件定义的函数
        if(!pFundetect){
                qDebug()<<"Get function detect failed"<<endl;
                return -1;
            }
//        PyObject_CallFunction(pFundetect,NULL);
        PyObject* args=PyTuple_New(1);
        QFile file_chuan("/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png");
        if(file_chuan.exists()){
            path = "/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png";
        }
        PyTuple_SetItem(args,0,Py_BuildValue("s",path.toStdString().c_str()));
        PyEval_CallObject(pFundetect, args);
//        Py_Finalize();

        QString show_path="./result_data/single_result_unet.jpg";
        ui->lbl_Image2->setPixmap(QPixmap(show_path));
        ui->lbl_Image2->setScaledContents(true);
    });
    //Unet test
    connect(ui->pushButton_3,&QPushButton::clicked,[=](){
        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
        Py_Initialize();
        if ( !Py_IsInitialized() )
        {
        return -1;
        }
        PyRun_SimpleString("import sys");//设置py脚本的路径
        PyRun_SimpleString("sys.path.append('./')");//当前路径
        PyObject* pModule = PyImport_ImportModule("eval_2");  // 这里的temp就是创建的python文件
        if (!pModule) {
                qDebug()<< "Cant open python file!\n" << endl;
                return -1;
        }
        PyObject* pFunhello = PyObject_GetAttrString(pModule,"fenge");  // 这里的hellow就是python文件定义的函数
        if(!pFunhello){
            qDebug()<<"Get function hello failed"<<endl;
            return -1;
        }
//            PyObject_CallFunction(pFunhello,NULL);
//        PyEval_CallObject(pFunhello, NULL);
        PyObject* pReturn = PyEval_CallObject(pFunhello, NULL);
        PyObject* x_1 = PyList_GetItem(pReturn, 0);
        PyObject* x_2 = PyList_GetItem(pReturn, 1);
        PyObject* x_3 = PyList_GetItem(pReturn, 2);
        out_1  = PyFloat_AsDouble(x_1);
        out_2  = PyFloat_AsDouble(x_2);
        out_3  = PyFloat_AsDouble(x_3);
        qDebug() << out_1;
        qDebug() << out_2;
        qDebug() << out_3;

        Py_Finalize();


        QString str_jac = QString::number(out_1);
        QListWidgetItem * item_jac = new QListWidgetItem("iou:"+str_jac);
        ui->listWidget->addItem(item_jac);
        item_jac->setTextAlignment(Qt::AlignHCenter);

        QString str_dsc = QString::number(out_2);
        QListWidgetItem * item_dsc = new QListWidgetItem("dsc:"+str_dsc);
        ui->listWidget->addItem(item_dsc);
        item_dsc->setTextAlignment(Qt::AlignHCenter);

        QString str_HD95 = QString::number(out_3);
        QListWidgetItem * item_HD95 = new QListWidgetItem("HD95:"+str_HD95);
        ui->listWidget->addItem(item_HD95);
        item_HD95->setTextAlignment(Qt::AlignHCenter);

    });
    //segformer test
    connect(ui->pushButton_seg_test,&QPushButton::clicked,[=](){
        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
        Py_Initialize();
        if ( !Py_IsInitialized() )
        {
        return -1;
        }
        PyRun_SimpleString("import sys");//设置py脚本的路径
        PyRun_SimpleString("sys.path.append('./')");//当前路径
        PyObject* pModule = PyImport_ImportModule("test_segformer");  // 这里的temp就是创建的python文件
        if (!pModule) {
                qDebug()<< "Cant open python file!\n" << endl;
                return -1;
        }
        PyObject* pFunhello = PyObject_GetAttrString(pModule,"fenge");  // 这里的hellow就是python文件定义的函数
        if(!pFunhello){
            qDebug()<<"Get function hello failed"<<endl;
            return -1;
        }
//            PyObject_CallFunction(pFunhello,NULL);
//        PyEval_CallObject(pFunhello, NULL);
        PyObject* pReturn = PyEval_CallObject(pFunhello, NULL);
        PyObject* x_1 = PyList_GetItem(pReturn, 0);
        PyObject* x_2 = PyList_GetItem(pReturn, 1);
        PyObject* x_3 = PyList_GetItem(pReturn, 2);
        out_1  = PyFloat_AsDouble(x_1);
        out_2  = PyFloat_AsDouble(x_2);
        out_3  = PyFloat_AsDouble(x_3);
        qDebug() << out_1;
        qDebug() << out_2;
        qDebug() << out_3;

        Py_Finalize();


        QString str_jac = QString::number(out_1);
        QListWidgetItem * item_jac = new QListWidgetItem("iou:"+str_jac);
        ui->listWidget->addItem(item_jac);
        item_jac->setTextAlignment(Qt::AlignHCenter);

        QString str_dsc = QString::number(out_2);
        QListWidgetItem * item_dsc = new QListWidgetItem("dsc:"+str_dsc);
        ui->listWidget->addItem(item_dsc);
        item_dsc->setTextAlignment(Qt::AlignHCenter);

        QString str_HD95 = QString::number(out_3);
        QListWidgetItem * item_HD95 = new QListWidgetItem("dsc:"+str_HD95);
        ui->listWidget->addItem(item_HD95);
        item_HD95->setTextAlignment(Qt::AlignHCenter);

    });
    //UTNET test
    connect(ui->pushButton_UTNET_test,&QPushButton::clicked,[=](){
        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
        Py_Initialize();
        if ( !Py_IsInitialized() )
        {
        return -1;
        }
        PyRun_SimpleString("import sys");//设置py脚本的路径
        PyRun_SimpleString("sys.path.append('./')");//当前路径
        PyObject* pModule = PyImport_ImportModule("test_UTNET");  // 这里的temp就是创建的python文件
        if (!pModule) {
                qDebug()<< "Cant open python file!\n" << endl;
                return -1;
        }
        PyObject* pFunhello = PyObject_GetAttrString(pModule,"fenge");  // 这里的hellow就是python文件定义的函数
        if(!pFunhello){
            qDebug()<<"Get function hello failed"<<endl;
            return -1;
        }
//            PyObject_CallFunction(pFunhello,NULL);
//        PyEval_CallObject(pFunhello, NULL);
        PyObject* pReturn = PyEval_CallObject(pFunhello, NULL);
        PyObject* x_1 = PyList_GetItem(pReturn, 0);
        PyObject* x_2 = PyList_GetItem(pReturn, 1);
        PyObject* x_3 = PyList_GetItem(pReturn, 2);
        out_1  = PyFloat_AsDouble(x_1);
        out_2  = PyFloat_AsDouble(x_2);
        out_3  = PyFloat_AsDouble(x_3);
        qDebug() << out_1;
        qDebug() << out_2;
        qDebug() << out_3;

        Py_Finalize();


        QString str_jac = QString::number(out_1);
        QListWidgetItem * item_jac = new QListWidgetItem("iou:"+str_jac);
        ui->listWidget->addItem(item_jac);
        item_jac->setTextAlignment(Qt::AlignHCenter);

        QString str_dsc = QString::number(out_2);
        QListWidgetItem * item_dsc = new QListWidgetItem("dsc:"+str_dsc);
        ui->listWidget->addItem(item_dsc);
        item_dsc->setTextAlignment(Qt::AlignHCenter);

        QString str_HD95 = QString::number(out_3);
        QListWidgetItem * item_HD95 = new QListWidgetItem("dsc:"+str_HD95);
        ui->listWidget->addItem(item_HD95);
        item_HD95->setTextAlignment(Qt::AlignHCenter);

    });
    //TransUnet test
    connect(ui->pushButton_TransUnet_test,&QPushButton::clicked,[=](){
        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
        Py_Initialize();
        if ( !Py_IsInitialized() )
        {
        return -1;
        }
        PyRun_SimpleString("import sys");//设置py脚本的路径
        PyRun_SimpleString("sys.path.append('./')");//当前路径
        PyObject* pModule = PyImport_ImportModule("test_transUnet");  // 这里的temp就是创建的python文件
        if (!pModule) {
                qDebug()<< "Cant open python file!\n" << endl;
                return -1;
        }
        PyObject* pFunhello = PyObject_GetAttrString(pModule,"fenge");  // 这里的hellow就是python文件定义的函数
        if(!pFunhello){
            qDebug()<<"Get function hello failed"<<endl;
            return -1;
        }
//            PyObject_CallFunction(pFunhello,NULL);
//        PyEval_CallObject(pFunhello, NULL);
        PyObject* pReturn = PyEval_CallObject(pFunhello, NULL);
        PyObject* x_1 = PyList_GetItem(pReturn, 0);
        PyObject* x_2 = PyList_GetItem(pReturn, 1);
        PyObject* x_3 = PyList_GetItem(pReturn, 2);
        out_1  = PyFloat_AsDouble(x_1);
        out_2  = PyFloat_AsDouble(x_2);
        out_3  = PyFloat_AsDouble(x_3);
        qDebug() << out_1;
        qDebug() << out_2;
        qDebug() << out_3;

        Py_Finalize();


        QString str_jac = QString::number(out_1);
        QListWidgetItem * item_jac = new QListWidgetItem("iou:"+str_jac);
        ui->listWidget->addItem(item_jac);
        item_jac->setTextAlignment(Qt::AlignHCenter);

        QString str_dsc = QString::number(out_2);
        QListWidgetItem * item_dsc = new QListWidgetItem("dsc:"+str_dsc);
        ui->listWidget->addItem(item_dsc);
        item_dsc->setTextAlignment(Qt::AlignHCenter);

        QString str_HD95 = QString::number(out_3);
        QListWidgetItem * item_HD95 = new QListWidgetItem("dsc:"+str_HD95);
        ui->listWidget->addItem(item_HD95);
        item_HD95->setTextAlignment(Qt::AlignHCenter);

    });

    //BPAT-UNet test
    connect(ui->pushButton_BPATUNet_test,&QPushButton::clicked,[=](){
        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
        Py_Initialize();
        if ( !Py_IsInitialized() )
        {
        return -1;
        }
        PyRun_SimpleString("import sys");//设置py脚本的路径
        PyRun_SimpleString("sys.path.append('./')");//当前路径
        PyObject* pModule = PyImport_ImportModule("test_BPAT-UNet");  // 这里的temp就是创建的python文件
        if (!pModule) {
                qDebug()<< "Cant open python file!\n" << endl;
                return -1;
        }
        PyObject* pFunhello = PyObject_GetAttrString(pModule,"fenge");  // 这里的hellow就是python文件定义的函数
        if(!pFunhello){
            qDebug()<<"Get function hello failed"<<endl;
            return -1;
        }
//            PyObject_CallFunction(pFunhello,NULL);
//        PyEval_CallObject(pFunhello, NULL);
        PyObject* pReturn = PyEval_CallObject(pFunhello, NULL);
        PyObject* x_1 = PyList_GetItem(pReturn, 0);
        PyObject* x_2 = PyList_GetItem(pReturn, 1);
        PyObject* x_3 = PyList_GetItem(pReturn, 2);
        out_1  = PyFloat_AsDouble(x_1);
        out_2  = PyFloat_AsDouble(x_2);
        out_3  = PyFloat_AsDouble(x_3);
        qDebug() << out_1;
        qDebug() << out_2;
        qDebug() << out_3;

        Py_Finalize();


        QString str_jac = QString::number(out_1);
        QListWidgetItem * item_jac = new QListWidgetItem("iou:"+str_jac);
        ui->listWidget->addItem(item_jac);
        item_jac->setTextAlignment(Qt::AlignHCenter);

        QString str_dsc = QString::number(out_2);
        QListWidgetItem * item_dsc = new QListWidgetItem("dsc:"+str_dsc);
        ui->listWidget->addItem(item_dsc);
        item_dsc->setTextAlignment(Qt::AlignHCenter);

        QString str_HD95 = QString::number(out_3);
        QListWidgetItem * item_HD95 = new QListWidgetItem("dsc:"+str_HD95);
        ui->listWidget->addItem(item_HD95);
        item_HD95->setTextAlignment(Qt::AlignHCenter);

    });

    //segformer predict
    connect(ui->pushButton_segformer,&QPushButton::clicked,[=](){
//        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
//        Py_SetPath(L"/home/caichengjie/anaconda3/envs/mctrans:"
//                   "/home/caichengjie/anaconda3/envs/mctrans/lib/python3.7/site-packages");
         Py_Initialize();
//        import_array();
         if ( !Py_IsInitialized() )
         {
            return -1;
         }
         PyRun_SimpleString("import sys");//设置py脚本的路径
         PyRun_SimpleString("sys.path.append('./')");//当前路径

         PyObject* pModule_detect = PyImport_ImportModule("predict_segformer");  // 这里的temp就是创建的python文件
         if (!pModule_detect) {
            qDebug()<< "Cant open python_detect_segformer file!\n" << endl;
            return -1;
         }
         PyObject* pFundetect = PyObject_GetAttrString(pModule_detect,"detect_img_segformer");  // 这里的hellow就是python文件定义的函数
         if(!pFundetect){
            qDebug()<<"Get function detect_segformer failed"<<endl;
            return -1;
         }
//        PyObject_CallFunction(pFundetect,NULL);
         PyObject* args=PyTuple_New(1);
         QFile file_chuan("/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png");
         if(file_chuan.exists()){
            path = "/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png";
         }
         PyTuple_SetItem(args,0,Py_BuildValue("s",path.toStdString().c_str()));
         PyEval_CallObject(pFundetect, args);
//        Py_Finalize();

         QString show_path="./result_data/single_result_segformer.jpg";
         ui->lbl_Image2->setPixmap(QPixmap(show_path));
         ui->lbl_Image2->setScaledContents(true);

    });
    //UTNET predict
    connect(ui->pushButton_UTNET,&QPushButton::clicked,[=](){
//        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
//        Py_SetPath(L"/home/caichengjie/anaconda3/envs/mctrans:"
//                   "/home/caichengjie/anaconda3/envs/mctrans/lib/python3.7/site-packages");
         Py_Initialize();
//        import_array();
         if ( !Py_IsInitialized() )
         {
            return -1;
         }
         PyRun_SimpleString("import sys");//设置py脚本的路径
         PyRun_SimpleString("sys.path.append('./')");//当前路径

         PyObject* pModule_detect = PyImport_ImportModule("predict_UTNET");  // 这里的temp就是创建的python文件
         if (!pModule_detect) {
            qDebug()<< "Cant open python_detect_UTNET file!\n" << endl;
            return -1;
         }
         PyObject* pFundetect = PyObject_GetAttrString(pModule_detect,"detect_img_UTNET");  // 这里的hellow就是python文件定义的函数
         if(!pFundetect){
            qDebug()<<"Get function detect_UTNET failed"<<endl;
            return -1;
         }
//        PyObject_CallFunction(pFundetect,NULL);
         PyObject* args=PyTuple_New(1);
         QFile file_chuan("/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png");
         if(file_chuan.exists()){
            path = "/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png";
         }
         PyTuple_SetItem(args,0,Py_BuildValue("s",path.toStdString().c_str()));
         PyEval_CallObject(pFundetect, args);
//        Py_Finalize();

         QString show_path="./result_data/single_result_UTNET.jpg";
         ui->lbl_Image2->setPixmap(QPixmap(show_path));
         ui->lbl_Image2->setScaledContents(true);

    });
    //TransUnet predict
    connect(ui->pushButton_TransUnet,&QPushButton::clicked,[=](){
//        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
//        Py_SetPath(L"/home/caichengjie/anaconda3/envs/mctrans:"
//                   "/home/caichengjie/anaconda3/envs/mctrans/lib/python3.7/site-packages");
         Py_Initialize();
//        import_array();
         if ( !Py_IsInitialized() )
         {
            return -1;
         }
         PyRun_SimpleString("import sys");//设置py脚本的路径
         PyRun_SimpleString("sys.path.append('./')");//当前路径

         PyObject* pModule_detect = PyImport_ImportModule("predict_transUnet");  // 这里的temp就是创建的python文件
         if (!pModule_detect) {
            qDebug()<< "Cant open python_detect_TransUnet file!\n" << endl;
            return -1;
         }
         PyObject* pFundetect = PyObject_GetAttrString(pModule_detect,"detect_img_TransUnet");  // 这里的hellow就是python文件定义的函数
         if(!pFundetect){
            qDebug()<<"Get function detect_TransUnet failed"<<endl;
            return -1;
         }
//        PyObject_CallFunction(pFundetect,NULL);
         PyObject* args=PyTuple_New(1);
         QFile file_chuan("/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png");
         if(file_chuan.exists()){
            path = "/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png";
         }
         PyTuple_SetItem(args,0,Py_BuildValue("s",path.toStdString().c_str()));
         PyEval_CallObject(pFundetect, args);
//        Py_Finalize();

         QString show_path="./result_data/single_result_TransUnet.jpg";
         ui->lbl_Image2->setPixmap(QPixmap(show_path));
         ui->lbl_Image2->setScaledContents(true);

    });

    //BPAT-UNet predict
    connect(ui->pushButton_BPATUNet,&QPushButton::clicked,[=](){
//        Py_SetPythonHome((const wchar_t *)(L"/home/caichengjie/anaconda3/envs/mctrans"));
//        Py_SetPath(L"/home/caichengjie/anaconda3/envs/mctrans:"
//                   "/home/caichengjie/anaconda3/envs/mctrans/lib/python3.7/site-packages");
         Py_Initialize();
//        import_array();
         if ( !Py_IsInitialized() )
         {
            return -1;
         }
         PyRun_SimpleString("import sys");//设置py脚本的路径
         PyRun_SimpleString("sys.path.append('./')");//当前路径

         PyObject* pModule_detect = PyImport_ImportModule("predict_BPAT-UNet");  // 这里的temp就是创建的python文件
         if (!pModule_detect) {
            qDebug()<< "Cant open python_detect_TransUnet file!\n" << endl;
            return -1;
         }
         PyObject* pFundetect = PyObject_GetAttrString(pModule_detect,"detect_img_BPATUNet");  // 这里的hellow就是python文件定义的函数
         if(!pFundetect){
            qDebug()<<"Get function detect_TransUnet failed"<<endl;
            return -1;
         }
//        PyObject_CallFunction(pFundetect,NULL);
         PyObject* args=PyTuple_New(1);
         QFile file_chuan("/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png");
         if(file_chuan.exists()){
            path = "/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png";
         }
         PyTuple_SetItem(args,0,Py_BuildValue("s",path.toStdString().c_str()));
         PyEval_CallObject(pFundetect, args);
//        Py_Finalize();

         QString show_path="./result_data/single_result_BPAT-UNet.jpg";
         ui->lbl_Image2->setPixmap(QPixmap(show_path));
         ui->lbl_Image2->setScaledContents(true);

    });


//    //original connect
//    totalBytes = 0;
//    bytesReceived = 0;
//    fileNameSize = 0;
//    tcpServer = new QTcpServer(this);
//    if(!tcpServer->listen(QHostAddress::Any,8989))
//    {  //**本地主机的6666端口，如果出错就输出错误信息，并关闭
//        qDebug() << tcpServer->errorString();
//        close();
//    }
//    //连接信号和相应槽函数,有新的连接进入是需处理
//    connect(tcpServer,SIGNAL(newConnection()),this,SLOT(NewConnection()));


    //new connection
    qDebug() << "服务器主线程: " << QThread::currentThread();

    m_s = new QTcpServer(this);

    // 当有客户端连接到来时， QTcpServer对象会发出newConnection信号
    connect(m_s, &QTcpServer::newConnection, this, [=]()
    {
        // 得到用于通信的套接字对象
        QTcpSocket* tcp = m_s->nextPendingConnection();
        // 创建子线程， 并且将用于通信的套接字对象传参给这个类的构造函数
        RecvFile* subThread = new RecvFile(tcp);
        subThread->start();

        // 捕捉over信号
        connect(subThread, &RecvFile::over, this, [=]()
        {
            subThread->exit();
            subThread->wait();
            subThread->deleteLater();
            QMessageBox::information(this, "文件接收", "文件接收完毕!!!");
            QString show_path_ori="/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/recv.png";
            ui->lbl_Image1->setPixmap(QPixmap(show_path_ori));
        });
    });
}

zhu::~zhu()
{
    delete ui;
}
//new listen
void zhu::on_setListen_clicked()
{
    unsigned short port = ui->port->text().toUShort();
    m_s->listen(QHostAddress::Any, port);
}

// original newConnection

//void zhu::NewConnection()
//{
//    //新连接进入的显示处理
//    currentClient = tcpServer->nextPendingConnection();
//    ui->label_wait->setText(tr("%1:%2").arg(currentClient->peerAddress().toString().split("::ffff:")[1])\
//            .arg(currentClient->peerPort()));
//    connect(currentClient, SIGNAL(readyRead()), this, SLOT(recMessage()));
//    connect(currentClient, SIGNAL(disconnected()), this, SLOT(disconnect()));

//}

//original receive

//void zhu::recMessage()
//{
//    QDataStream in(currentClient);
//    in.setVersion(QDataStream::Qt_5_8);
//    if(bytesReceived <= sizeof(qint64)*2)
//    { //如果接收到的数据小于16个字节，那么是刚开始接收数据，我们保存到//来的头文件信息

//        if((currentClient->bytesAvailable() >= sizeof(qint64)*2)

//                && (fileNameSize == 0))

//        { //接收数据总大小信息和文件名大小信息

//            in >> totalBytes >> fileNameSize;

//            bytesReceived += sizeof(qint64) * 2;

//        }

//        if((currentClient->bytesAvailable() >= fileNameSize)

//                && (fileNameSize != 0))

//        {  //接收文件名，并建立文件

//            in >> fileName;

//            ui->label_wait->setText(tr("接收文件 %1 ...").arg(fileName));

//            bytesReceived += fileNameSize;
//            ui->label_wait->setText(fileName);
//            localFile= new QFile(fileName);
//            if(!localFile->open(QFile::WriteOnly))
//            {
//                qDebug() << "open file error!";
//                return;
//            }
//        }
//        else return;
//    }
//    if(bytesReceived < totalBytes)
//    {  //如果接收的数据小于总数据，那么写入文件
//        bytesReceived += currentClient->bytesAvailable();
//        inBlock+= currentClient->readAll();
//    }
//    //更新进度条
//    ui->progressBar->setMaximum(totalBytes);
//    ui->progressBar->setValue(bytesReceived);
//    if(bytesReceived == totalBytes)
//    { //接收数据完成时
//        //接收显示
//        QBuffer buffer(&inBlock);
//        buffer.open(QIODevice::ReadOnly);
//        QImageReader reader(&buffer,"png");
//        QImage image = reader.read();
//        if(!image.isNull())
//        {
//            image.save("/home/caichengjie/QTproject/login/build-login-Desktop_Qt_5_12_0_GCC_64bit-Debug/data/tn3k_dian_2/cunchu/rec.png");
//            image=image.scaled(ui->lbl_Image1->size());
//            ui->lbl_Image1->setPixmap(QPixmap::fromImage(image));
//        }
//        localFile->write(inBlock);
//        localFile->close();
//        inBlock.resize(0);
//        //重新置0 准备下次接收
//        totalBytes = 0;
//        bytesReceived = 0;
//        fileNameSize = 0;
//        ui->label_wait->setText(tr("接收文件 %1 成功！").arg(fileName));
//    }
//}


//original disconnect

//void zhu::disconnect()
//{
//    qDebug()<<"disconnect";
//}



