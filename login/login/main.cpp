#include "widget.h"
#include <QApplication>
#include <QFile>
#include <QTextStream>
float out_1;
float out_2;
float out_3;
QString path;
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);


    Widget w;
    w.show();

    return app.exec();
}
