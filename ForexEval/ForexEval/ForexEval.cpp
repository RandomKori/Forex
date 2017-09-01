// ForexEval.cpp: определяет экспортированные функции для приложения DLL.
//

#include "stdafx.h"

extern "C"  __declspec(dllexport) void LoadModel(char* s);
extern "C" __declspec(dllexport) void EvalMoswl(double inp[], double out[]);

using namespace Microsoft::MSR::CNTK;


Eval<double>* model;

void LoadModel(char* s)
{
	std::string config("deviceId=auto modelPath=.\\Models\\");
	config.append(s);
	config.append(" minibatchSize=1024");
	model = new Eval<double>(config);
}

void EvalMoswl(double inp[], double out[])
{
	std::vector<double> v(45);
	for (int i = 0; i < 45; i++)
		v[i] = inp[i];
	std::wstring s = L"features";
	std::map<std::wstring, std::vector<double>*> inpt;
	inpt.insert(std::map<std::wstring, std::vector<double>*>::value_type(s, &v));
	std::vector<double> v1(3);
	std::wstring s1 = L"labwls";
	std::map<std::wstring, std::vector<double>*> outt;
	outt.insert(std::map<std::wstring, std::vector<double>*>::value_type(s1, &v1));
	model->Evaluate(inpt, outt);
	for (int i = 0; i < 3; i++)
		out[i] = v1[i];
}


