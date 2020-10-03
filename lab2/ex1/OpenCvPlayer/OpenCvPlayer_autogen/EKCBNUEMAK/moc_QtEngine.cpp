/****************************************************************************
** Meta object code from reading C++ file 'QtEngine.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.12.8)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "Engine/QtEngine.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QtEngine.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.12.8. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_player_engine__Engine_t {
    QByteArrayData data[13];
    char stringdata0[153];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_player_engine__Engine_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_player_engine__Engine_t qt_meta_stringdata_player_engine__Engine = {
    {
QT_MOC_LITERAL(0, 0, 21), // "player_engine::Engine"
QT_MOC_LITERAL(1, 22, 10), // "changeTime"
QT_MOC_LITERAL(2, 33, 0), // ""
QT_MOC_LITERAL(3, 34, 5), // "frame"
QT_MOC_LITERAL(4, 40, 11), // "changeSpeed"
QT_MOC_LITERAL(5, 52, 5), // "speed"
QT_MOC_LITERAL(6, 58, 15), // "loopStateChange"
QT_MOC_LITERAL(7, 74, 4), // "loop"
QT_MOC_LITERAL(8, 79, 19), // "sequencePauseToggle"
QT_MOC_LITERAL(9, 99, 17), // "sequenceNextFrame"
QT_MOC_LITERAL(10, 117, 17), // "sequencePrevFrame"
QT_MOC_LITERAL(11, 135, 11), // "setInputDir"
QT_MOC_LITERAL(12, 147, 5) // "about"

    },
    "player_engine::Engine\0changeTime\0\0"
    "frame\0changeSpeed\0speed\0loopStateChange\0"
    "loop\0sequencePauseToggle\0sequenceNextFrame\0"
    "sequencePrevFrame\0setInputDir\0about"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_player_engine__Engine[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    1,   54,    2, 0x08 /* Private */,
       4,    1,   57,    2, 0x08 /* Private */,
       6,    1,   60,    2, 0x08 /* Private */,
       8,    0,   63,    2, 0x08 /* Private */,
       9,    0,   64,    2, 0x08 /* Private */,
      10,    0,   65,    2, 0x08 /* Private */,
      11,    0,   66,    2, 0x08 /* Private */,
      12,    0,   67,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void, QMetaType::Int,    3,
    QMetaType::Void, QMetaType::Int,    5,
    QMetaType::Void, QMetaType::Int,    7,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void player_engine::Engine::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<Engine *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->changeTime((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: _t->changeSpeed((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: _t->loopStateChange((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->sequencePauseToggle(); break;
        case 4: _t->sequenceNextFrame(); break;
        case 5: _t->sequencePrevFrame(); break;
        case 6: _t->setInputDir(); break;
        case 7: _t->about(); break;
        default: ;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject player_engine::Engine::staticMetaObject = { {
    &QMainWindow::staticMetaObject,
    qt_meta_stringdata_player_engine__Engine.data,
    qt_meta_data_player_engine__Engine,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *player_engine::Engine::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *player_engine::Engine::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_player_engine__Engine.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int player_engine::Engine::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 8)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 8;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 8)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 8;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
