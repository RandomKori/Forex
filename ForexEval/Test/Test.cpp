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
	auto var1 = InputVariable(NDShape({ 45,1 }), DataType::Float, L"features");
	auto var2 = InputVariable(NDShape({ 3,1 }), DataType::Float, L"labels");
	inps = Value::CreateBatch(NDShape({ 45,1 }), v, d);
	//outs = Value::CreateBatch(NDShape({ 3,1 }), v1, d);
	std::unordered_map<Variable, ValuePtr> inputLayer = { { var1, inps } };
	std::unordered_map<Variable, ValuePtr> outputLayer = { { var2, outs } };
	model->Evaluate(inputLayer, outputLayer);
	return 0;
}

