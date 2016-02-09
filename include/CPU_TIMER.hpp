#ifndef __TIMER_HPP__
#define __TIMER_HPP__


#include<vector>
#include<ctime>

/**
 A timer for CPU code.
 It measures time between start() and stop().
*/
class TIMER
{

  private:   

    clock_t _clocks; 
    double  _elapsed_millisecs; 

  public:

    TIMER() : _clocks(0), _elapsed_millisecs(0) {};

    /** It starts the timer. */
    void start()
    {
      _clocks = std::clock();
    }

    /** It stops the timer. */
    void stop()
    {
      _clocks = std::clock() - _clocks;
      _elapsed_millisecs = 1000 * double(_clocks) / CLOCKS_PER_SEC;
    }

    /** It returns the milliseconds between start() and stop() */
    double elapsed_millisecs() 
    {
      return _elapsed_millisecs;
    }

};

#endif

