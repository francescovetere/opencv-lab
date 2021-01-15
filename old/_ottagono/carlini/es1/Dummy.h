#ifndef OPENCVPLAYER_H
#define OPENCVPLAYER_H

/**
 * \file dummy.cpp
 * \author VisLab (vislab@ce.unipr.it)
 * \date 2016-09-21
 */


#include "Engine/QtEngine.h"

class Dummy : public player_engine::Engine
{
  public:
      Dummy(){};
      
  private:
      virtual void On_Execute() override;
    
};

#endif // OpenCvPlayer_H
