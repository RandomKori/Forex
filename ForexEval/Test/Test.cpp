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
	const std::vector<float> v1(3);
	std::vector<std::vector<float>> v2(1,v1);
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
	try
	{
		model->Evaluate(inputLayer, outputLayer);
		auto outs1 = outputLayer[var2];
		outs1->CopyVariableValueTo(var2, v2);
	}
	catch (const std::exception& e)
	{
		int g = 0;
	}
	return 0;
}

