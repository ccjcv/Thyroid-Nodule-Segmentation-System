#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include "zhu.h"

namespace Ui {
class Widget;
}

class Widget : public QWidget
{
    Q_OBJECT

public:
    explicit Widget(QWidget *parent = 0);
    ~Widget();
    bool connectDB();

    zhu *ppage2 = NULL;




private:
    Ui::Widget *ui;
};

#endif // WIDGET_H
