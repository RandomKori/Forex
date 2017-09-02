// ForexEval.cpp: определяет экспортированные функции для приложения DLL.
//

#include "stdafx.h"

using namespace CNTK;

extern "C" __declspec(dllexport) void LoadModel(wchar_t* s);
extern "C" __declspec(dllexport) void EvalModel(double* inp, double* out);

FunctionPtr model;

void LoadModel(wchar_t* s)
{
	model = Function::Load(s);
}

void EvalModel(double* inp, double* out)
{
	std::vector<double> v(45);
	std::vector<double> v1(3);
	for (int i = 0; i < 45; i++)
		v[i] = inp[i];
	ValuePtr inps;
	ValuePtr outs;
	DeviceDescriptor d= DeviceDescriptor::UseDefaultDevice();
	NDShape sp = {45};
	inps = Value::CreateSequence(sp,v,d);
	NDShape sp1 = {3};
	outs = Value::CreateSequence(sp1, v1, d);
	auto var1= InputVariable(NDShape({ 45 }), DataType::Double, { Axis::DefaultDynamicAxis() });
	auto var2 = InputVariable(NDShape({ 3 }), DataType::Double, { Axis::DefaultBatchAxis() });
	std::unordered_map<Variable, ValuePtr> inputLayer;
	std::unordered_map<Variable, ValuePtr> outputLayer;
	inputLayer.insert(std::pair<Variable, ValuePtr>(var1, inps));
	outputLayer.insert(std::pair<Variable, ValuePtr>(var2, outs));
	model->Evaluate(inputLayer, outputLayer);
	for (int i = 0; i < 3; i++)
		out[i] = v1[i];
}


