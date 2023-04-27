#include "widget.h"
#include "ui_widget.h"
#include <QFile>
#include <QTextStream>
#include "zhu.h"
#include <QPushButton>
#include <QDebug>
#include <QSqlDatabase>
#include <QMessageBox>
#include <QSqlError>
#include <QSqlQuery>
#include <QComboBox>
#include <QSqlQueryModel>

Widget::Widget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Widget)
{
    ui->setupUi(this);
    ui->label_user_name->setScaledContents(true);   //图片自适应label大小
    ui->label_pwd->setScaledContents(true);         //图片自适应label大小

    ui->lineE_pwd->setEchoMode(QLineEdit::Password);//设置为小黑点

    QString filePath;
    filePath = ":/res/qss/style-4.qss";

    /*皮肤设置*/
    QFile file(filePath);/*QSS文件所在的路径*/
    file.open(QFile::ReadOnly);
    QTextStream filetext(&file);
    QString stylesheet = filetext.readAll();
    this->setStyleSheet(stylesheet);
    file.close();
    
    connectDB();
    this->ppage2=new zhu;
    connect(ui->btn_login,&QPushButton::clicked,[=](){

        QSqlQuery query1;
        query1.exec("SELECT * FROM user");
        QVariantList userlista;
        QString usert;
        usert=ui->lineE_user_name->text();
        QString passwdt;
        passwdt=ui->lineE_pwd->text();
        bool T1=false;
        while(query1.next()){
            qDebug()<<query1.value(0).toString();
            qDebug()<<query1.value(1).toString();
            if(query1.value(0).toString()==usert&&query1.value(1).toString()==passwdt){
                T1=true;
            }
        }
        qDebug()<<T1;
        if(T1==true){
            QMessageBox::information(this,"success","Successfully login");
            this->hide();
            this->ppage2->show();
        }
        else{
            QMessageBox::information(this,"warning","somgthing is wrong");
        }
        query1.execBatch();


    });

    connect(this->ppage2,&zhu::back,[=](){
        this->ppage2->hide();
        this->show();
    });    
}

bool Widget::connectDB(){
    qDebug() <<  QSqlDatabase::drivers();
    //tianjia mysql shujuku
    QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
    //lianjie shujuku
    db.setHostName("localhost");
    db.setUserName("root");
    db.setPassword("123456");
    db.setDatabaseName("bh");
    if(!db.open()){
        QMessageBox::critical(this,"mysql fails to open",db.lastError().text());
        return false;
    }
    else{
        return true;
    }
}

Widget::~Widget()
{
    delete ui;
}
