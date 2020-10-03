#ifndef QT_ENGINE_H
#define QT_ENGINE_H

/**
 * \author Domenico Giaquinto - dgiaquinto@ambarella.com
 * \date 21 September 2016
 * 
 */

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdio.h>
#include <thread>
#include <chrono>

#include <QMainWindow>
#include <QAction>
#include <QCheckBox>
#include <QLabel>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace player_engine { 

  class Engine : public QMainWindow
  {
    Q_OBJECT
    public:
	Engine();
	virtual ~Engine();
	
	enum Cameras{Right,Left};
	
	//! Load the folder with images.
	void Load(const char *dirname);
	//! Returns the path of the frame (Right or Left if stereo, mono if it is not necessary to specify the camera) available.
	/**
	  * Returns an empty string if the camera does not exist.
	  * It is recommended to check if the path is valid.
	  * E.g. First Method
	  * cv::Mat src = cv::imread(getFrameFromCamera(Right));
	  *
	  * if(! src.data )// Check for invalid input
	  *  {
	  *    std::cout <<  "Could not open or find the image" << std::endl ;
	  *    return ;
	  *  }
	  * 
	  *  cv::namedWindow("App",CV_WINDOW_NORMAL);
	  *  cv::imshow("App", src);
	  * 
	  *  E.g. Second Method
	  *  std::string frame = cv::imread(getFrameFromCamera(Right));
	  *  if(frame.empty())
	  *  {
	  *   std::cout <<  "Could not open or find the image" << std::endl ;
	  *   return ;
	  *  }
	  *  
	  *  cv::namedWindow("App",CV_WINDOW_NORMAL);
	  *  cv::imshow("App", src);
	  */
	const std::string & getFrameFromCamera(const Cameras camera = Right);
	
    protected:
	virtual void On_Execute() = 0;
	
    private:
 	struct Synchronize {
	  enum States{Init, Run, Next ,Prev , Loop, Skip};
	  void reset (unsigned int max = 0){
	    nframeL 	    = 0;
	    nframeR 	    = 0;
	    nframeCurrent   = 0;
	    nframeSequences = max;
	    state 	    = Synchronize::Init;
	  }
	  
	  States state = Synchronize::Init;
	  unsigned int nframeL 		= 0;
	  unsigned int nframeR 		= 0;
	  unsigned int nframeCurrent 	= 0;
	  unsigned int nframeSequences	= 0;
	  
	};
	
	void CreateCommandMenu();
	void CreateMenuBar();
	void Run();
	void incrementFrameNumber();
	const unsigned int getSyncroFrame();
	const unsigned int getSpeed(); 
	
	QAction	*m_playPauseSequencesAct; 
	QAction	*m_prevSequencesAct;
	QAction	*m_nextSequencesAct;
	QAction *m_inputDir; 
	QAction *m_loadDirMenu;
	
	QVBoxLayout *m_mainLayout;
	
	QSlider   *m_TimeSlider;
	QSpinBox  *m_TimeSpinBox;
	QSpinBox  *m_SpeedSpinBox;  
	QCheckBox *m_LoopFrameCheckBox;
	QCheckBox *m_CameraLeft;
	QCheckBox *m_CameraRight;
	QCheckBox *m_EnableApp;
	QLabel	  *m_LabelInfo;
	QLabel	  *m_LabelnFrame;
	
	    
	Synchronize m_synchronizer;
	
	std::string m_folder;
	
	std::vector<std::string> *m_RightVec;
	std::vector<std::string> *m_LeftVec;
	
	std::thread m_thx;
	std::atomic<bool> m_notClosing;
	std::mutex m_mtx; 
	std::condition_variable m_cvWakeUp;
	
	bool m_paused;
	unsigned int m_speed;
	std::mutex m_graphicParameters;
	
	std::chrono::time_point<std::chrono::high_resolution_clock> m_lastTime;
	
	
	
    private Q_SLOTS:
      void changeTime(int frame);
      void changeSpeed(int speed);
      void loopStateChange(int loop);
      void sequencePauseToggle();
      void sequenceNextFrame();
      void sequencePrevFrame();
      void setInputDir();
      void about();
      
  };
}

#endif 