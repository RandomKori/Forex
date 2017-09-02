// ForexEval.cpp: определяет экспортированные функции для приложения DLL.
//

#include "stdafx.h"

extern "C"  __declspec(dllexport) void __stdcall LoadModel(wchar_t* s);
extern "C" __declspec(dllexport) void __stdcall EvalModel(double* inp, double* out);

using namespace CNTK;

FunctionPtr model;

void __stdcall LoadModel(wchar_t* s)
{
	std::wstring config=L"./Models/";
	config.append(s);
	model->Load(config);
}

void __stdcall EvalModel(double* inp, double* out)
{
	std::vector<double> v(45);
	std::vector<double> v1(3);
	for (int i = 0; i < 45; i++)
		v[i] = inp[i];
	ValuePtr inps;
	ValuePtr outs;
	DeviceDescriptor d= DeviceDescriptor::UseDefaultDevice();
	NDShape sp(45,1);
	inps = Value::CreateSequence(sp,v,d);
	NDShape sp1(3, 1);
	outs = Value::CreateSequence(sp1, v1, d);
	Variable var1;
	var1= InputVariable({ 45 }, AsDataType<double>(), L"features");
	Variable var2;
	var2 = InputVariable({ 3 }, AsDataType<double>(), L"labels");
	std::unordered_map<Variable, ValuePtr> inputLayer;
	std::unordered_map<Variable, ValuePtr> outputLayer;
	inputLayer.insert(std::pair<Variable, ValuePtr>(var1, inps));
	outputLayer.insert(std::pair<Variable, ValuePtr>(var2, outs));
	model->Evaluate(inputLayer, outputLayer);
	for (int i = 0; i < 3; i++)
		out[i] = v1[i];
}


