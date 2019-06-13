/*
Author: Yashoteja Prabhu ( yashoteja.prabhu@gmail.com )
*/


#pragma once

#include <iostream>
#include <chrono>

#include "config.h"
#include "utils.h"

using namespace std;
using namespace std::chrono;


/* ----------------------------- timer resource ------------------------------------ */

class Timer
{
	private:
		_float sum_time = 0;
		high_resolution_clock::time_point start_time;
		high_resolution_clock::time_point stop_time;

	public:
		Timer()
		{
			sum_time = 0;
		}

		void start()    // starts fresh time measurement
		{
			sum_time = 0;
			start_time = high_resolution_clock::now();
		}

		void resume()   // resumes time measurement, adding to the up-to-now elapsed time
		{
			start_time = high_resolution_clock::now();
		}

		_float stop()    // returns elapsed time in seconds  
		{
			stop_time = high_resolution_clock::now();
			sum_time += duration_cast<microseconds>(stop_time - start_time).count() / 1000000.0;
			return sum_time;
		}
};

