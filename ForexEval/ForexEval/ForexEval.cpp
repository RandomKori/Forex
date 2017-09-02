// ForexEval.cpp: определяет экспортированные функции для приложения DLL.
//

#include "stdafx.h"

extern "C"  __declspec(dllexport) void __stdcall LoadModel(wchar_t* s);
extern "C" __declspec(dllexport) void __stdcall EvalModel(double* inp, double* out);

using namespace Microsoft::MSR::CNTK;


IEvaluateModel<double>* model;

void __stdcall LoadModel(wchar_t* s)
{
	std::string config("deviceId=auto minibatchSize=1024 modelPath=\"./Models/");
	char s1[256];
	size_t len=std::wcstombs(s1,s, wcslen(s));
	if (len > 0u)
		s1[len] = '\0';
	config.append(s1);
	config.append("\"");
	GetEvalD(&model);
	model->Init(config);
}

void __stdcall EvalModel(double* inp, double* out)
{
	std::vector<double> v(45);
	for (int i = 0; i < 45; i++)
		v[i] = inp[i];
	std::wstring s = L"features";
	std::map<std::wstring, std::vector<double>*> inpt;
	inpt.insert(std::map<std::wstring, std::vector<double>*>::value_type(s, &v));
	std::vector<double> v1(3);
	std::wstring s1 = L"labels";
	std::map<std::wstring, std::vector<double>*> outt;
	outt.insert(std::map<std::wstring, std::vector<double>*>::value_type(s1, &v1));
	model->Evaluate(inpt, outt);
	for (int i = 0; i < 3; i++)
		out[i] = v1[i];
}


