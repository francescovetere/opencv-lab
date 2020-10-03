#include "QtEngine.h"

/**
 * \author Domenico Giaquinto - dgiaquinto@ambarella.com
 * \date 21 September 2016
 * 
 */

#include "Engine/paths.h"

#include <dirent.h>
#include <iostream>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <QDockWidget>
#include <QFileDialog>
#include <QIcon>
#include <QMenu>
#include <QMenuBar>
#include <QPushButton>
#include <QString>
#include <QStyle>
#include <QToolBar>
#include <QMessageBox>



using namespace player_engine;

Engine::Engine() 
	  :m_playPauseSequencesAct(new QAction(this))
	  , m_prevSequencesAct(new QAction(this))
	  , m_nextSequencesAct(new QAction(this))
	  , m_inputDir(new QAction(this))
	  , m_loadDirMenu(new QAction(this))
	  , m_mainLayout(new QVBoxLayout())
	  , m_RightVec (new std::vector<std::string>)
	  , m_LeftVec  (new std::vector<std::string>)
	  , m_notClosing(true)
	  , m_paused(true)
	  , m_speed(100)
{ 
    setAttribute(Qt::WA_DeleteOnClose);
    setWindowTitle(tr("Vislab Player V1.0"));
    setFixedSize(330,250);

    CreateMenuBar();
    CreateCommandMenu();
    
    QWidget* centralWidget = new QWidget();
    centralWidget->setLayout(m_mainLayout);
    setCentralWidget( centralWidget );
    
    m_lastTime = std::chrono::high_resolution_clock::now();
    m_thx = std::thread([this](){ Run(); });
}

Engine::~Engine()
{
  m_notClosing = false;
  {
    std::unique_lock<std::mutex> lock(m_mtx);
    m_paused = false;
  }
  m_cvWakeUp.notify_all();
  m_thx.join();
  cv::destroyAllWindows();
}

void Engine::Run(){
  while (m_notClosing) {
    {
      std::unique_lock<std::mutex> lock(m_mtx);
      while (m_paused) {
	m_cvWakeUp.wait(lock);
	m_lastTime = std::chrono::high_resolution_clock::now();
      }
    }
    if (!m_notClosing) break;
    incrementFrameNumber();
    if(m_EnableApp->isChecked()) {
      On_Execute();
    } else {
      std::string a = getFrameFromCamera();
    }
     
    const std::chrono::high_resolution_clock::time_point endTime = std::chrono::high_resolution_clock::now();
    const std::chrono::high_resolution_clock::time_point nextTime = m_lastTime+std::chrono::milliseconds(getSpeed());
    m_lastTime = nextTime;
    if (nextTime > endTime) {
      std::this_thread::sleep_until(nextTime);
    }
   }
}

void Engine::incrementFrameNumber()
{ 
  std::unique_lock<std::mutex> lock(m_graphicParameters); 
  
  switch (m_synchronizer.state){
    case Synchronize::Prev:
	if(m_synchronizer.nframeCurrent != 0){
	  --m_synchronizer.nframeCurrent;
	}
	m_synchronizer.state = Synchronize::Run;
	{
	  std::unique_lock<std::mutex> lock(m_mtx);
	  m_paused = true;
	}
      break;
    case Synchronize::Next:
	if(m_synchronizer.nframeCurrent < m_synchronizer.nframeSequences - 1)
	    ++m_synchronizer.nframeCurrent;
	m_synchronizer.state = Synchronize::Run;
	{
	  std::unique_lock<std::mutex> lock(m_mtx);
	  m_paused = true;
	}
      break;
    case Synchronize::Loop:
      break;
    case Synchronize::Init:
	m_synchronizer.state = Synchronize::Run;
      break;
    case Synchronize::Skip:
	m_synchronizer.state = Synchronize::Run;
	{
	  std::unique_lock<std::mutex> lock(m_mtx);
	  m_paused = true;
	}
      break;
    default:
	if(m_synchronizer.nframeCurrent < m_synchronizer.nframeSequences - 1)
	  ++m_synchronizer.nframeCurrent;
	m_synchronizer.state = Synchronize::Run;
  } 

  if( (m_synchronizer.nframeCurrent == m_synchronizer.nframeSequences - 1)  && (!m_LoopFrameCheckBox->isChecked()) ){
    QMetaObject::invokeMethod( this, "sequencePauseToggle");
  }
}


void Engine::CreateMenuBar()
{
  QMenu* fileMenu = menuBar()->addMenu(tr("&File") );
  {
    m_loadDirMenu->setText( "Load" );
    connect(m_loadDirMenu, SIGNAL(triggered()), SLOT(setInputDir()) );
    fileMenu->addAction( m_loadDirMenu );
    
    QAction* quit = new QAction(this);
    quit->setText( "Quit" );
    connect(quit, SIGNAL(triggered()), SLOT(close()) );
    fileMenu->addAction( quit );
  }
  menuBar()->addSeparator();
  QMenu* helpMenu = menuBar()->addMenu(tr("&Help"));
  {
    QAction* aboutAct = new QAction(tr("&About"),this);
    connect(aboutAct, SIGNAL(triggered()), SLOT(about()));
    helpMenu->addAction(aboutAct);
  }
  
  
}

void Engine::CreateCommandMenu()
{
  m_EnableApp 	= new QCheckBox("&App",this);
  m_EnableApp->setEnabled(true);
  m_EnableApp->setChecked(true);
  
  m_CameraLeft  = new QCheckBox("&Left",this);
  m_CameraRight = new QCheckBox("&Right",this);
  m_CameraLeft ->setEnabled(false);
  m_CameraRight->setEnabled(false);
  
  QLabel * labelCameras = new QLabel();
  labelCameras->setText("   Cameras Avaiable");
  
  QHBoxLayout * hLayoutImage = new QHBoxLayout();
  hLayoutImage->addWidget(m_EnableApp);
  hLayoutImage->addWidget(labelCameras);
  hLayoutImage->addWidget(m_CameraLeft);
  hLayoutImage->addWidget(m_CameraRight);
  hLayoutImage->setStretch(1,1);
  hLayoutImage->setStretch(2,1);
  hLayoutImage->setStretch(3,7);
  
  QWidget * widgetImage = new QWidget();
  widgetImage->setLayout(hLayoutImage);
  
  m_mainLayout->addWidget(widgetImage);
  
  QToolBar * mainToolBar = addToolBar("Vislab Player");
  mainToolBar->setMovable(false);
  
  m_LoopFrameCheckBox = new QCheckBox("&Loop",this);
  m_LoopFrameCheckBox->setEnabled(false);
  connect(m_LoopFrameCheckBox,SIGNAL(stateChanged(int)),this,SLOT(loopStateChange(int)));
  mainToolBar->addWidget(m_LoopFrameCheckBox);

  m_playPauseSequencesAct->setText("&Start");
  m_playPauseSequencesAct->setIcon(QIcon::fromTheme("media-playback-start",this->style()->standardIcon(QStyle::SP_MediaPlay)));
  m_playPauseSequencesAct->setEnabled(false);
  m_playPauseSequencesAct->setStatusTip("PAUSE");
  connect(m_playPauseSequencesAct,SIGNAL(triggered()),this,SLOT(sequencePauseToggle()));
  mainToolBar->addAction(m_playPauseSequencesAct);
  
  m_inputDir = new QAction(tr("&Input Directory: "),this);
  m_inputDir->setIcon(QIcon::fromTheme("media-playback-open",this->style()->standardIcon(QStyle::SP_DirOpenIcon)));
  connect(m_inputDir, SIGNAL(triggered()), SLOT(setInputDir()));
  mainToolBar->addAction(m_inputDir);
  
  m_prevSequencesAct->setText("&Prev");
  m_prevSequencesAct->setIcon(QIcon::fromTheme("media-playback-left",this->style()->standardIcon(QStyle::SP_ArrowLeft)));
  m_prevSequencesAct->setEnabled(false);
  connect(m_prevSequencesAct, SIGNAL(triggered()), SLOT(sequencePrevFrame()));
  mainToolBar->addAction(m_prevSequencesAct);
  
  m_nextSequencesAct->setText("&Next");
  m_nextSequencesAct->setIcon(QIcon::fromTheme("media-playback-right",this->style()->standardIcon(QStyle::SP_ArrowRight)));
  m_nextSequencesAct->setEnabled(false);
  connect(m_nextSequencesAct, SIGNAL(triggered()), SLOT(sequenceNextFrame()));
  mainToolBar->addAction(m_nextSequencesAct);
  
  QHBoxLayout * hWaitBox = new QHBoxLayout();
  
  QLabel * labelSpinBox = new QLabel();
  labelSpinBox->setText("Wait between two frames [ms]");
  hWaitBox->addWidget(labelSpinBox);
  hWaitBox->setStretch(0,9);
  
  m_SpeedSpinBox = new QSpinBox();
  m_SpeedSpinBox->setRange(1,1000);
  m_SpeedSpinBox->setValue(m_speed);
  connect(m_SpeedSpinBox, SIGNAL(valueChanged(int)), SLOT(changeSpeed(int)));
  hWaitBox->addWidget(m_SpeedSpinBox);
  hWaitBox->setStretch(0,1);
  
  QWidget * widgetWaitBox = new QWidget();
  widgetWaitBox->setLayout(hWaitBox);
  
  m_mainLayout->addWidget(widgetWaitBox);
  
  m_mainLayout->addWidget(mainToolBar);
  
  QHBoxLayout * hLayout = new QHBoxLayout();
  
  QLabel * labelFrame = new QLabel();
  labelFrame->setText("Frame n.");
  hLayout->addWidget(labelFrame);
  
  m_LabelnFrame = new QLabel();
  m_LabelnFrame->setText("0");
  m_LabelnFrame->setMinimumWidth(35);
  m_LabelnFrame->setMaximumWidth(35);
  hLayout->addWidget(m_LabelnFrame);
  
  m_TimeSlider = new QSlider(Qt::Horizontal);
  m_TimeSlider->setRange(0,0);
  m_TimeSlider->setSingleStep(1);
  connect(m_TimeSlider, SIGNAL(valueChanged(int)), SLOT(changeTime(int)));
  hLayout->addWidget(m_TimeSlider);
  
  QWidget * widgetTime = new QWidget();
  widgetTime->setLayout(hLayout);
  
  m_mainLayout->addWidget(widgetTime);
  
  m_LabelInfo = new QLabel();
  m_LabelInfo->setAlignment(Qt::AlignCenter);
  
  m_mainLayout->addWidget(m_LabelInfo);
  
}

void Engine::about() {
  QMessageBox::about(this, tr("About VisLab Player v1.0"), tr("<center> VisLab Player v1.0 \n is a tool designed specifically for student, \n for more information ask \n Domenico Giaquinto (giaq@vislab.it)</center>"));
}


void Engine::sequencePauseToggle()
{
  std::unique_lock<std::mutex> lock(m_mtx);
  if(m_playPauseSequencesAct->statusTip()=="PAUSE"){
    m_paused = false;
    m_playPauseSequencesAct->setText("Pause packets list");
    m_playPauseSequencesAct->setIcon(QIcon::fromTheme("media-playback-pause",this->style()->standardIcon(QStyle::SP_MediaPause)));
    m_playPauseSequencesAct->setStatusTip("PLAY");
    m_prevSequencesAct->setEnabled(false);
    m_nextSequencesAct->setEnabled(false);
    m_inputDir->setEnabled(false);
    m_loadDirMenu->setEnabled(false);
    m_cvWakeUp.notify_all();
  }
  else{
    m_playPauseSequencesAct->setText("Start");
    m_playPauseSequencesAct->setIcon(QIcon::fromTheme("media-playback-start",this->style()->standardIcon(QStyle::SP_MediaPlay)));
    m_playPauseSequencesAct->setStatusTip("PAUSE");
    if(!m_LoopFrameCheckBox->isChecked()){
      m_prevSequencesAct->setEnabled(true);
      m_nextSequencesAct->setEnabled(true);
      m_inputDir->setEnabled(true);
      m_loadDirMenu->setEnabled(true);
    }
    m_paused = true;
  }
}

void Engine::sequenceNextFrame()
{
  {
    std::unique_lock<std::mutex> lock(m_graphicParameters);
    m_synchronizer.state = Synchronize::Next;
  }
  std::unique_lock<std::mutex> lock(m_mtx);
  {
    m_paused = false;
  }
  m_cvWakeUp.notify_all();
}

void Engine::sequencePrevFrame()
{
  {
    std::unique_lock<std::mutex> lock(m_graphicParameters);
    m_synchronizer.state = Synchronize::Prev;
  }
  std::unique_lock<std::mutex> lock(m_mtx);
  {
    m_paused = false;
  }
  m_cvWakeUp.notify_all();
}

void Engine::changeTime(const int frame)
{
  std::string  playPauseSequencesAct;
  {
    std::unique_lock<std::mutex> lock(m_graphicParameters);
    m_LabelnFrame->setText(QString::number(frame));
    m_synchronizer.nframeCurrent = frame;
    playPauseSequencesAct  = m_playPauseSequencesAct->statusTip().toStdString();
    
  }
  
  if(playPauseSequencesAct == "PAUSE"){	
    {
      std::unique_lock<std::mutex> lock(m_graphicParameters);
      m_synchronizer.state = Synchronize::Skip;
    }
    std::unique_lock<std::mutex> lock(m_mtx);
    {
      m_paused = false;
    }
  }
  m_cvWakeUp.notify_all();
}

void Engine::changeSpeed(int speed)
{
  std::unique_lock<std::mutex> lock(m_graphicParameters);
  m_speed = speed;
}

void Engine::loopStateChange(const int loop)
{
  std::unique_lock<std::mutex> lock(m_graphicParameters); 
  if(loop){
    m_prevSequencesAct->setEnabled(false);
    m_nextSequencesAct->setEnabled(false);
    m_inputDir->setEnabled(false);
    m_loadDirMenu->setEnabled(false);
    m_synchronizer.state = Synchronize::Loop;
  }
  else{
    m_synchronizer.state = Synchronize::Run;
    if(m_playPauseSequencesAct->statusTip() == "PAUSE"){
      m_prevSequencesAct->setEnabled(true);
      m_nextSequencesAct->setEnabled(true);
      m_inputDir->setEnabled(true);
      m_loadDirMenu->setEnabled(true);
    }
  }
}


void Engine::setInputDir()
{ 
  QString src = QFileDialog::getExistingDirectory(this,tr("Choose Or Create Directory"),(QDir::homePath()).append("/"),QFileDialog::DontResolveSymlinks);
  
  if(!src.isNull()) Load((src+"/").toLatin1().data());
  
}

void Engine::Load(const char *dirname) {
  std::unique_lock<std::mutex> lock(m_graphicParameters);
  
  m_playPauseSequencesAct->setEnabled(false);
  m_prevSequencesAct	 ->setEnabled(false);
  m_nextSequencesAct	 ->setEnabled(false);
  m_LoopFrameCheckBox    ->setEnabled(false);
  m_CameraRight		 ->setEnabled(false);
  m_CameraLeft 		 ->setEnabled(false);
  
  m_TimeSlider->blockSignals(true);
  m_TimeSlider->setValue(0);
  m_TimeSlider->setRange(0,0);
  m_TimeSlider->blockSignals(false);
  m_LabelnFrame->setText("0");
    
  m_LeftVec->clear();
  m_RightVec->clear();
  m_synchronizer.reset();
    
  if(!dirname){    
      m_LabelInfo->setStyleSheet("QLabel { color : red; }");
      m_LabelInfo->setText("INFO:: Load a Directory");
      return;
    }
   
  DIR *dp(opendir(dirname));
  dirent *d = new dirent();
  bool validDirectory = false;
  std::string value;

  while((d = readdir(dp)) != NULL){
    value = static_cast<std::string>(d->d_name);    
    if(value.find(".pgm")!=std::string::npos || value.find(".ppm")!=std::string::npos || value.find(".jpg")!=std::string::npos){
      validDirectory = true;
      if (value.find("L.")!=std::string::npos)
	m_LeftVec->push_back(d->d_name);
      else
	m_RightVec->push_back(d->d_name);
      }
  }

  if(!validDirectory || ( ((m_RightVec->size() > 0) && (m_LeftVec->size() > 0)) && (m_RightVec->size() != m_LeftVec->size()))){
    m_LabelInfo->setStyleSheet("QLabel { color : red; }");
    m_LabelInfo->setText("INFO:: The directory doesn't contain valid file!");
    return;
  }
  
  m_synchronizer.reset(std::max(m_RightVec->size(), m_LeftVec->size())); 
  m_TimeSlider->setRange(0,(m_synchronizer.nframeSequences>0)?m_synchronizer.nframeSequences-1:0);
  
  std::sort(m_RightVec->begin(), m_RightVec->end());
  std::sort(m_LeftVec ->begin(), m_LeftVec ->end());
  
    
  m_folder = static_cast<std::string>(dirname);
  
  m_LabelInfo->setStyleSheet("QLabel { color : blue; }");
  m_LabelInfo->setText("INFO:: Directory Loaded");
  
  m_playPauseSequencesAct->setEnabled(true);
  m_prevSequencesAct	 ->setEnabled(true);
  m_nextSequencesAct	 ->setEnabled(true);
  m_LoopFrameCheckBox	 ->setEnabled(true);
  
  if(m_RightVec->size() > 0)m_CameraRight->setEnabled(true);
  if(m_LeftVec ->size() > 0)m_CameraLeft ->setEnabled(true);
        
}

const std::string & Engine::getFrameFromCamera(const Cameras camera)
{
  std::string * frame = new  std::string();
  unsigned int nFrame = 0;
  nFrame = getSyncroFrame();
  
  if(m_CameraRight->isChecked()){
    cv::namedWindow("RIGHT",cv::WINDOW_NORMAL);
    cv::imshow("RIGHT", cv::imread((m_folder + (*m_RightVec)[nFrame])));
  }
  if(m_CameraLeft->isChecked()){
    cv::namedWindow("LEFT",cv::WINDOW_NORMAL);
    cv::imshow("LEFT", cv::imread((m_folder + (*m_LeftVec)[nFrame])));
  }
  
  if (camera == Right){
    if(m_RightVec->size() != 0 ){
      frame = new std::string((m_folder + (*m_RightVec)[nFrame]));
    }
  }else{
    if(m_LeftVec->size() != 0 ){
      frame  = new std::string((m_folder + (*m_LeftVec)[nFrame]));
    }
  }
  
  {
    std::unique_lock<std::mutex> lock(m_graphicParameters);
    m_TimeSlider->blockSignals(true);
    m_TimeSlider->setValue(nFrame);
    m_TimeSlider->blockSignals(false);
    m_LabelnFrame->setText(QString::number(nFrame));
  }
  
  if(frame->empty()){
    std::cerr << "\033[1;31mERROR:: CAMERA NOT AVAILABLE.\033[0m\n";
    frame = new std::string(ERROR_CONFIG);
  }
  
  return *frame;
}

const unsigned int Engine::getSyncroFrame()
{
  unsigned int nframeCurrent;
  Synchronize::States state;
  {
    std::unique_lock<std::mutex> lock(m_graphicParameters);
    state = m_synchronizer.state;
    nframeCurrent = m_synchronizer.nframeCurrent;
  }
  
  if(state == Synchronize::Prev || state == Synchronize::Next){
    std::unique_lock<std::mutex> lock(m_mtx);
    m_paused = true;
  }
   return nframeCurrent;
}

const unsigned int Engine::getSpeed()
{
  std::unique_lock<std::mutex> lock(m_graphicParameters); 
  return m_speed;
}

