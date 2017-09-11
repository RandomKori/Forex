// ForexEvalHL1.cpp: определяет экспортированные функции для приложения DLL.
//

#include "stdafx.h"
#include "CNTKLibrary.h"

using namespace CNTK;
using namespace std;

extern "C" __declspec(dllexport) void LoadModel(wchar_t* s);
extern "C" __declspec(dllexport) void EvalModel(double* s, double* h, double* l, double* v, double* out);

FunctionPtr model;

void LoadModel(wchar_t* s)
{
	const wstring s1(s);
	model = Function::Load(s1);
}

void EvalModel(double* s, double* h, double* l, double* v, double* out)
{
	std::vector<float> vs(10);
	std::vector<float> vh(10);
	std::vector<float> vl(10);
	std::vector<float> vv(10);
	const std::vector<float> v1(2);
	std::vector<std::vector<float>> v2(1, v1);
	for (int i = 0; i < 10; i++)
	{
		vs[i] = (float)(s[i]);
		vh[i] = (float)(h[i]);
		vl[i] = (float)(l[i]);
		vv[i] = (float)(v[i]);
	}
	ValuePtr inps;
	ValuePtr inph;
	ValuePtr inpl;
	ValuePtr inpv;
	ValuePtr outs;
	DeviceDescriptor d = DeviceDescriptor::UseDefaultDevice();
	auto var1 = model->Arguments();
	auto var2 = model->Output();
	inps = Value::CreateBatch(NDShape({ 10 }), vs, d, false);
	inph = Value::CreateBatch(NDShape({ 10 }), vh, d, false);
	inpl = Value::CreateBatch(NDShape({ 10 }), vl, d, false);
	inpv = Value::CreateBatch(NDShape({ 10 }), vv, d, false);
	std::unordered_map<Variable, ValuePtr> inputLayer = { { var1[0], inps}, {var1[1], inph}, {var1[2], inpl}, {var1[3], inpv }};
	std::unordered_map<Variable, ValuePtr> outputLayer = { { var2, outs } };
	model->Evaluate(inputLayer, outputLayer);
	auto outs1 = outputLayer[var2];
	outs1->CopyVariableValueTo(var2, v2);
	for (int i = 0; i < 2; i++)
		out[i] = v2[0][i];
}


