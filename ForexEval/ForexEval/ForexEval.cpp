// ForexEval.cpp: определяет экспортированные функции для приложения DLL.
//

#include "stdafx.h"

using namespace CNTK;
using namespace std;

extern "C" __declspec(dllexport) void LoadModel(wchar_t* s);
extern "C" __declspec(dllexport) void EvalModel(double* inp, double* out);

FunctionPtr model;

void LoadModel(wchar_t* s)
{
	const wstring s1(s);
	model = Function::Load(s1);
}

void EvalModel(double* inp, double* out)
{
	std::vector<float> v(45);
	std::vector<float> v1(3);
	for (int i = 0; i < 45; i++)
		v[i] = 0.5;
	ValuePtr inps;
	ValuePtr outs;
	DeviceDescriptor d = DeviceDescriptor::UseDefaultDevice();
	auto var1 = model->Arguments();
	auto var2 = model->Output();
	inps = Value::CreateBatch(NDShape({ 45 }), v, d, false);
	std::unordered_map<Variable, ValuePtr> inputLayer = { { var1[0], inps } };
	std::unordered_map<Variable, ValuePtr> outputLayer = { { var2, outs } };
	model->Evaluate(inputLayer, outputLayer);
}


