// Test.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <string>



using namespace CNTK;
using namespace std;

int main()
{
	const wstring s(L"D:\\Models\\model.cmf");
	FunctionPtr model;
	try
	{
		model = Function::Load(s);
	}
	catch (const std::exception&)
	{

	}
	std::vector<float> v(45);
	std::vector <float> v1(3);
	for (int i = 0; i < 45; i++)
		v[i] = 0.5;
	ValuePtr inps;
	ValuePtr outs;
	DeviceDescriptor d = DeviceDescriptor::UseDefaultDevice();
	auto var1 = model->Inputs();
	inps = Value::CreateBatch(NDShape({ 45 }), v, d);
	outs = Value::CreateBatch(NDShape({ 3 }), v1, d);
	std::unordered_map<Variable, ValuePtr> inputLayer = { { var1[0], inps } };
	std::unordered_map<Variable, ValuePtr> outputLayer = { { var1[1], outs } };
	try
	{
		model->Evaluate(inputLayer, outputLayer);
	}
	catch (const std::exception& e)
	{
		int g = 0;
	}
	return 0;
}

