#pragma once

#include <Windows.h>

class Timer
{
private:
	bool __runningFlag = false;

	LARGE_INTEGER __tFreq, __tStart, __tEnd;
	float __elapsedTime = 0.f;

public:
	Timer() { QueryPerformanceFrequency(&__tFreq); };
	Timer(const Timer&) = delete;
	Timer(Timer&&) = delete;

	Timer& operator=(const Timer&) = delete;
	Timer operator=(Timer&&) = delete;

	inline bool Start() {
		if (!__runningFlag)
		{
			QueryPerformanceCounter(&__tStart);
			__runningFlag = true;

			return true;
		}
		return false;
	};
	inline bool End() {
		if (__runningFlag)
		{
			QueryPerformanceCounter(&__tEnd);
			__elapsedTime = (((__tEnd.QuadPart - __tStart.QuadPart) / static_cast<float>(__tFreq.QuadPart)) * 1000.f); //ms second

			if (__elapsedTime < 0.f)
				__elapsedTime = 0.f;

			__runningFlag = false;

			return true;
		}

		return false;
	};

	inline const float GetElapsedTime() const { return __elapsedTime; };
};

