#ifndef TIMER_H
#define TIMER_H

#include <ctime>
#include <iostream>
#include <iomanip>
#include "util.h"

class Timer
{
  friend std::ostream& operator<<(std::ostream& os, Timer& t);

private:
  bool running;
  time_t start_time;

  //TODO in seconds?
  double elapsed_time();

public:
  /***
   * 'running' is initially false.  A timer needs to be explicitly started
   * using 'start' or 'restart'
   */
  Timer() : running(false), start_time(0) { }

  void start(const char* msg = 0);
//  void restart(const char* msg = 0);
//  void stop(const char* msg = 0);
  void check(const char* msg = 0);
  double get_elapsed_time();

};

#endif // TIMER_H
