// Test.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"

extern "C"  __declspec(dllimport) void LoadModel(wchar_t* s);
extern "C" __declspec(dllimport) void EvalModel(double* inp, double* out);

int main()
{
	wchar_t* s= L"model.cmf";
	LoadModel(s);
	return 0;
}

