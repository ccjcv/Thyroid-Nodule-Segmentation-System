/********************************************************************************
** Form generated from reading UI file 'zhu.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ZHU_H
#define UI_ZHU_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_zhu
{
public:
    QHBoxLayout *horizontalLayout_7;
    QFrame *frame_background;
    QPushButton *btn_huizhu;
    QLabel *lbl_title;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_3;
    QWidget *widget_4;
    QGridLayout *gridLayout;
    QWidget *widget_6;
    QHBoxLayout *horizontalLayout_3;
    QSpacerItem *horizontalSpacer_4;
    QPushButton *pushButton_2;
    QSpacerItem *horizontalSpacer_10;
    QPushButton *pushButton_segformer;
    QSpacerItem *horizontalSpacer_5;
    QWidget *widget_14;
    QHBoxLayout *horizontalLayout_11;
    QSpacerItem *horizontalSpacer_11;
    QPushButton *pushButton_UTNET;
    QSpacerItem *horizontalSpacer_13;
    QPushButton *pushButton_TransUnet;
    QSpacerItem *horizontalSpacer_12;
    QWidget *widget_10;
    QHBoxLayout *horizontalLayout_8;
    QLabel *label;
    QLineEdit *port;
    QPushButton *setListen;
    QLabel *lbl_Image1;
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QLabel *label_pre;
    QWidget *widget_11;
    QVBoxLayout *verticalLayout_2;
    QWidget *widget_3;
    QVBoxLayout *verticalLayout_3;
    QLineEdit *lineEdit;
    QPushButton *pushButton;
    QLabel *lbl_Image2;
    QWidget *widget_5;
    QWidget *widget_12;
    QGridLayout *gridLayout_4;
    QPushButton *pushButton_BPATUNet;
    QSpacerItem *verticalSpacer;
    QFrame *frame_tu1;
    QWidget *widget_13;
    QHBoxLayout *horizontalLayout_10;
    QWidget *widget_2;
    QGridLayout *gridLayout_2;
    QWidget *widget_9;
    QHBoxLayout *horizontalLayout_5;
    QWidget *widget_7;
    QHBoxLayout *horizontalLayout_4;
    QSpacerItem *horizontalSpacer_9;
    QListWidget *listWidget;
    QSpacerItem *horizontalSpacer_8;
    QWidget *widget_8;
    QVBoxLayout *verticalLayout;
    QWidget *widget_17;
    QHBoxLayout *horizontalLayout_2;
    QLabel *lbl_test;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton_BPATUNet_test;
    QWidget *widget_16;
    QHBoxLayout *horizontalLayout_6;
    QPushButton *pushButton_seg_test;
    QPushButton *pushButton_3;
    QWidget *widget_15;
    QHBoxLayout *horizontalLayout_12;
    QPushButton *pushButton_UTNET_test;
    QPushButton *pushButton_TransUnet_test;

    void setupUi(QWidget *zhu)
    {
        if (zhu->objectName().isEmpty())
            zhu->setObjectName(QString::fromUtf8("zhu"));
        zhu->resize(1024, 668);
        horizontalLayout_7 = new QHBoxLayout(zhu);
        horizontalLayout_7->setObjectName(QString::fromUtf8("horizontalLayout_7"));
        frame_background = new QFrame(zhu);
        frame_background->setObjectName(QString::fromUtf8("frame_background"));
        frame_background->setFrameShape(QFrame::StyledPanel);
        frame_background->setFrameShadow(QFrame::Raised);
        btn_huizhu = new QPushButton(frame_background);
        btn_huizhu->setObjectName(QString::fromUtf8("btn_huizhu"));
        btn_huizhu->setGeometry(QRect(920, 10, 80, 25));
        QFont font;
        font.setPointSize(12);
        btn_huizhu->setFont(font);
        lbl_title = new QLabel(frame_background);
        lbl_title->setObjectName(QString::fromUtf8("lbl_title"));
        lbl_title->setGeometry(QRect(20, 10, 891, 41));
        QFont font1;
        font1.setFamily(QString::fromUtf8("Noto Sans Devanagari UI"));
        font1.setPointSize(25);
        font1.setBold(true);
        font1.setWeight(75);
        lbl_title->setFont(font1);
        groupBox = new QGroupBox(frame_background);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(20, 60, 501, 561));
        gridLayout_3 = new QGridLayout(groupBox);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        widget_4 = new QWidget(groupBox);
        widget_4->setObjectName(QString::fromUtf8("widget_4"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(widget_4->sizePolicy().hasHeightForWidth());
        widget_4->setSizePolicy(sizePolicy);
        gridLayout = new QGridLayout(widget_4);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        widget_6 = new QWidget(widget_4);
        widget_6->setObjectName(QString::fromUtf8("widget_6"));
        sizePolicy.setHeightForWidth(widget_6->sizePolicy().hasHeightForWidth());
        widget_6->setSizePolicy(sizePolicy);
        horizontalLayout_3 = new QHBoxLayout(widget_6);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_4);

        pushButton_2 = new QPushButton(widget_6);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        QFont font2;
        font2.setFamily(QString::fromUtf8("Noto Sans Gurmukhi"));
        font2.setPointSize(12);
        pushButton_2->setFont(font2);

        horizontalLayout_3->addWidget(pushButton_2);

        horizontalSpacer_10 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_10);

        pushButton_segformer = new QPushButton(widget_6);
        pushButton_segformer->setObjectName(QString::fromUtf8("pushButton_segformer"));
        pushButton_segformer->setFont(font2);

        horizontalLayout_3->addWidget(pushButton_segformer);

        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_3->addItem(horizontalSpacer_5);


        gridLayout->addWidget(widget_6, 1, 0, 1, 1);

        widget_14 = new QWidget(widget_4);
        widget_14->setObjectName(QString::fromUtf8("widget_14"));
        horizontalLayout_11 = new QHBoxLayout(widget_14);
        horizontalLayout_11->setObjectName(QString::fromUtf8("horizontalLayout_11"));
        horizontalSpacer_11 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_11);

        pushButton_UTNET = new QPushButton(widget_14);
        pushButton_UTNET->setObjectName(QString::fromUtf8("pushButton_UTNET"));
        pushButton_UTNET->setFont(font);

        horizontalLayout_11->addWidget(pushButton_UTNET);

        horizontalSpacer_13 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_13);

        pushButton_TransUnet = new QPushButton(widget_14);
        pushButton_TransUnet->setObjectName(QString::fromUtf8("pushButton_TransUnet"));
        pushButton_TransUnet->setFont(font);

        horizontalLayout_11->addWidget(pushButton_TransUnet);

        horizontalSpacer_12 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_11->addItem(horizontalSpacer_12);


        gridLayout->addWidget(widget_14, 2, 0, 1, 1);


        gridLayout_3->addWidget(widget_4, 2, 0, 1, 2);

        widget_10 = new QWidget(groupBox);
        widget_10->setObjectName(QString::fromUtf8("widget_10"));
        horizontalLayout_8 = new QHBoxLayout(widget_10);
        horizontalLayout_8->setObjectName(QString::fromUtf8("horizontalLayout_8"));
        label = new QLabel(widget_10);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_8->addWidget(label);

        port = new QLineEdit(widget_10);
        port->setObjectName(QString::fromUtf8("port"));

        horizontalLayout_8->addWidget(port);

        setListen = new QPushButton(widget_10);
        setListen->setObjectName(QString::fromUtf8("setListen"));

        horizontalLayout_8->addWidget(setListen);


        gridLayout_3->addWidget(widget_10, 0, 1, 1, 1);

        lbl_Image1 = new QLabel(groupBox);
        lbl_Image1->setObjectName(QString::fromUtf8("lbl_Image1"));
        lbl_Image1->setMinimumSize(QSize(200, 200));
        lbl_Image1->setMaximumSize(QSize(200, 200));
        QFont font3;
        font3.setFamily(QString::fromUtf8("Noto Sans Arabic UI"));
        font3.setPointSize(14);
        lbl_Image1->setFont(font3);
        lbl_Image1->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(lbl_Image1, 3, 0, 1, 1);

        widget = new QWidget(groupBox);
        widget->setObjectName(QString::fromUtf8("widget"));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_pre = new QLabel(widget);
        label_pre->setObjectName(QString::fromUtf8("label_pre"));
        sizePolicy.setHeightForWidth(label_pre->sizePolicy().hasHeightForWidth());
        label_pre->setSizePolicy(sizePolicy);
        QFont font4;
        font4.setFamily(QString::fromUtf8("Noto Sans Adlam"));
        font4.setPointSize(14);
        font4.setBold(true);
        font4.setWeight(75);
        label_pre->setFont(font4);
        label_pre->setAlignment(Qt::AlignCenter);

        horizontalLayout->addWidget(label_pre);


        gridLayout_3->addWidget(widget, 0, 0, 1, 1);

        widget_11 = new QWidget(groupBox);
        widget_11->setObjectName(QString::fromUtf8("widget_11"));
        verticalLayout_2 = new QVBoxLayout(widget_11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        widget_3 = new QWidget(widget_11);
        widget_3->setObjectName(QString::fromUtf8("widget_3"));
        sizePolicy.setHeightForWidth(widget_3->sizePolicy().hasHeightForWidth());
        widget_3->setSizePolicy(sizePolicy);
        verticalLayout_3 = new QVBoxLayout(widget_3);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        lineEdit = new QLineEdit(widget_3);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        verticalLayout_3->addWidget(lineEdit);

        pushButton = new QPushButton(widget_3);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setFont(font2);

        verticalLayout_3->addWidget(pushButton);


        verticalLayout_2->addWidget(widget_3);


        gridLayout_3->addWidget(widget_11, 1, 0, 1, 1);

        lbl_Image2 = new QLabel(groupBox);
        lbl_Image2->setObjectName(QString::fromUtf8("lbl_Image2"));
        lbl_Image2->setMinimumSize(QSize(200, 200));
        lbl_Image2->setMaximumSize(QSize(200, 200));
        QFont font5;
        font5.setFamily(QString::fromUtf8("Noto Sans Telugu"));
        font5.setPointSize(14);
        lbl_Image2->setFont(font5);
        lbl_Image2->setAlignment(Qt::AlignCenter);

        gridLayout_3->addWidget(lbl_Image2, 3, 1, 1, 1);

        widget_5 = new QWidget(groupBox);
        widget_5->setObjectName(QString::fromUtf8("widget_5"));
        widget_12 = new QWidget(widget_5);
        widget_12->setObjectName(QString::fromUtf8("widget_12"));
        widget_12->setGeometry(QRect(0, 0, 241, 101));
        gridLayout_4 = new QGridLayout(widget_12);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        pushButton_BPATUNet = new QPushButton(widget_12);
        pushButton_BPATUNet->setObjectName(QString::fromUtf8("pushButton_BPATUNet"));
        QFont font6;
        font6.setPointSize(12);
        font6.setBold(true);
        font6.setWeight(75);
        pushButton_BPATUNet->setFont(font6);

        gridLayout_4->addWidget(pushButton_BPATUNet, 1, 0, 2, 2);

        verticalSpacer = new QSpacerItem(20, 64, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_4->addItem(verticalSpacer, 0, 1, 1, 1);


        gridLayout_3->addWidget(widget_5, 1, 1, 1, 1);

        frame_tu1 = new QFrame(frame_background);
        frame_tu1->setObjectName(QString::fromUtf8("frame_tu1"));
        frame_tu1->setGeometry(QRect(530, 60, 471, 231));
        frame_tu1->setFrameShape(QFrame::StyledPanel);
        frame_tu1->setFrameShadow(QFrame::Raised);
        widget_13 = new QWidget(frame_tu1);
        widget_13->setObjectName(QString::fromUtf8("widget_13"));
        widget_13->setGeometry(QRect(290, 220, 70, 43));
        horizontalLayout_10 = new QHBoxLayout(widget_13);
        horizontalLayout_10->setObjectName(QString::fromUtf8("horizontalLayout_10"));
        widget_2 = new QWidget(frame_background);
        widget_2->setObjectName(QString::fromUtf8("widget_2"));
        widget_2->setGeometry(QRect(560, 280, 401, 361));
        gridLayout_2 = new QGridLayout(widget_2);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        widget_9 = new QWidget(widget_2);
        widget_9->setObjectName(QString::fromUtf8("widget_9"));
        horizontalLayout_5 = new QHBoxLayout(widget_9);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        widget_7 = new QWidget(widget_9);
        widget_7->setObjectName(QString::fromUtf8("widget_7"));
        horizontalLayout_4 = new QHBoxLayout(widget_7);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        horizontalSpacer_9 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_9);

        listWidget = new QListWidget(widget_7);
        listWidget->setObjectName(QString::fromUtf8("listWidget"));
        listWidget->setFont(font4);

        horizontalLayout_4->addWidget(listWidget);

        horizontalSpacer_8 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_4->addItem(horizontalSpacer_8);


        horizontalLayout_5->addWidget(widget_7);


        gridLayout_2->addWidget(widget_9, 2, 0, 1, 1);

        widget_8 = new QWidget(widget_2);
        widget_8->setObjectName(QString::fromUtf8("widget_8"));
        verticalLayout = new QVBoxLayout(widget_8);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        widget_17 = new QWidget(widget_8);
        widget_17->setObjectName(QString::fromUtf8("widget_17"));
        horizontalLayout_2 = new QHBoxLayout(widget_17);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        lbl_test = new QLabel(widget_17);
        lbl_test->setObjectName(QString::fromUtf8("lbl_test"));
        QFont font7;
        font7.setFamily(QString::fromUtf8("Noto Sans Adlam"));
        font7.setPointSize(16);
        font7.setBold(true);
        font7.setWeight(75);
        lbl_test->setFont(font7);
        lbl_test->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);

        horizontalLayout_2->addWidget(lbl_test);

        horizontalSpacer = new QSpacerItem(150, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer);

        pushButton_BPATUNet_test = new QPushButton(widget_17);
        pushButton_BPATUNet_test->setObjectName(QString::fromUtf8("pushButton_BPATUNet_test"));
        pushButton_BPATUNet_test->setFont(font6);

        horizontalLayout_2->addWidget(pushButton_BPATUNet_test);


        verticalLayout->addWidget(widget_17);

        widget_16 = new QWidget(widget_8);
        widget_16->setObjectName(QString::fromUtf8("widget_16"));
        horizontalLayout_6 = new QHBoxLayout(widget_16);
        horizontalLayout_6->setObjectName(QString::fromUtf8("horizontalLayout_6"));
        pushButton_seg_test = new QPushButton(widget_16);
        pushButton_seg_test->setObjectName(QString::fromUtf8("pushButton_seg_test"));
        pushButton_seg_test->setFont(font);

        horizontalLayout_6->addWidget(pushButton_seg_test);

        pushButton_3 = new QPushButton(widget_16);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));
        QFont font8;
        font8.setFamily(QString::fromUtf8("Noto Sans Hanunoo"));
        font8.setPointSize(12);
        pushButton_3->setFont(font8);

        horizontalLayout_6->addWidget(pushButton_3);


        verticalLayout->addWidget(widget_16);

        widget_15 = new QWidget(widget_8);
        widget_15->setObjectName(QString::fromUtf8("widget_15"));
        horizontalLayout_12 = new QHBoxLayout(widget_15);
        horizontalLayout_12->setObjectName(QString::fromUtf8("horizontalLayout_12"));
        pushButton_UTNET_test = new QPushButton(widget_15);
        pushButton_UTNET_test->setObjectName(QString::fromUtf8("pushButton_UTNET_test"));
        pushButton_UTNET_test->setFont(font);

        horizontalLayout_12->addWidget(pushButton_UTNET_test);

        pushButton_TransUnet_test = new QPushButton(widget_15);
        pushButton_TransUnet_test->setObjectName(QString::fromUtf8("pushButton_TransUnet_test"));
        pushButton_TransUnet_test->setFont(font);

        horizontalLayout_12->addWidget(pushButton_TransUnet_test);


        verticalLayout->addWidget(widget_15);


        gridLayout_2->addWidget(widget_8, 1, 0, 1, 1);


        horizontalLayout_7->addWidget(frame_background);


        retranslateUi(zhu);

        QMetaObject::connectSlotsByName(zhu);
    } // setupUi

    void retranslateUi(QWidget *zhu)
    {
        zhu->setWindowTitle(QApplication::translate("zhu", "Form", nullptr));
        btn_huizhu->setText(QApplication::translate("zhu", "return", nullptr));
        lbl_title->setText(QApplication::translate("zhu", "Ultrasonic thyroid nodule segmentation system", nullptr));
        groupBox->setTitle(QString());
        pushButton_2->setText(QApplication::translate("zhu", "Start Unet Segment", nullptr));
        pushButton_segformer->setText(QApplication::translate("zhu", "Start Segformer Segment", nullptr));
        pushButton_UTNET->setText(QApplication::translate("zhu", "Start UTNET Segment", nullptr));
        pushButton_TransUnet->setText(QApplication::translate("zhu", "Start TransUnet Segment", nullptr));
        label->setText(QApplication::translate("zhu", "port:", nullptr));
        setListen->setText(QApplication::translate("zhu", "start listen", nullptr));
        lbl_Image1->setText(QApplication::translate("zhu", "Original Image", nullptr));
        label_pre->setText(QApplication::translate("zhu", "Predict", nullptr));
        pushButton->setText(QApplication::translate("zhu", "open", nullptr));
        lbl_Image2->setText(QApplication::translate("zhu", "Segmentation Image", nullptr));
        pushButton_BPATUNet->setText(QApplication::translate("zhu", "Start BPAT-UNet Segment", nullptr));
        lbl_test->setText(QApplication::translate("zhu", "Test", nullptr));
        pushButton_BPATUNet_test->setText(QApplication::translate("zhu", "BPAT-UNet Test", nullptr));
        pushButton_seg_test->setText(QApplication::translate("zhu", "Segformer Test", nullptr));
        pushButton_3->setText(QApplication::translate("zhu", "Unet Test", nullptr));
        pushButton_UTNET_test->setText(QApplication::translate("zhu", "UTNET Test", nullptr));
        pushButton_TransUnet_test->setText(QApplication::translate("zhu", "TransUnet Test", nullptr));
    } // retranslateUi

};

namespace Ui {
    class zhu: public Ui_zhu {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ZHU_H
