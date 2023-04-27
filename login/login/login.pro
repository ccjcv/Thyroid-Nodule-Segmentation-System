#-------------------------------------------------
#
# Project created by QtCreator 2022-07-21T21:49:39
#
#-------------------------------------------------

QT       += core gui sql
QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = login
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        widget.cpp \
    zhu.cpp \
    recvfile.cpp

HEADERS += \
        widget.h \
    zhu.h \
    recvfile.h

FORMS += \
        widget.ui \
    zhu.ui

RESOURCES += \
    qss.qrc \
    pic.qrc

unix:!macx: LIBS += -L /home/caichengjie/anaconda3/envs/mctrans/lib/ -lpython3.7m

INCLUDEPATH += /home/caichengjie/anaconda3/envs/mctrans/include/python3.7m
DEPENDPATH += /home/caichengjie/anaconda3/envs/mctrans/include/python3.7m
INCLUDEPATH += /home/caichengjie/anaconda3/envs/mctrans/lib/python3.7/site-packages/
#INCLUDEPATH += /home/caichengjie/anaconda3/envs/mctrans/lib/python3.7/site-packages/numpy/core/include/

unix:!macx: PRE_TARGETDEPS += /home/caichengjie/anaconda3/envs/mctrans/lib/libpython3.7m.a
