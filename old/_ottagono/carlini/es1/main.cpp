/**
 * \file main.cpp
 * \author VisLab (vislab@ce.unipr.it)
 * \date 2016-09-21
 */

#include "Dummy.h"
#include <iostream>

#include <QApplication>

int main(int argc, char** argv)
{
    QApplication app(argc, argv);
    Dummy* foo = new Dummy();
    foo->show();
    
    foo->Load(argv[1]);
    
    const int r = app.exec();
    
    return r;    
}
