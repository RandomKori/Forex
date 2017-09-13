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
	std::vector<float> v(30);
	const std::vector<float> v1(4);
	std::vector<std::vector<float>> v2(1, v1);
	for (int i = 0; i < 30; i++)
		v[i] = (float)(inp[i]);
	ValuePtr inps;
	ValuePtr outs;
	DeviceDescriptor d = DeviceDescriptor::GPUDevice(0);
	auto var1 = model->Arguments();
	auto var2 = model->Output();
	inps = Value::CreateBatch(NDShape({ 30 }), v, d, false);
	std::unordered_map<Variable, ValuePtr> inputLayer = { { var1[0], inps } };
	std::unordered_map<Variable, ValuePtr> outputLayer = { { var2, outs } };
	model->Evaluate(inputLayer, outputLayer);
	auto outs1 = outputLayer[var2];
	outs1->CopyVariableValueTo(var2, v2);
	for (int i = 0; i < 4; i++)
		out[i] = v2[0][i];
}


