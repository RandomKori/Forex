// Test1.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include "Eval.h"

using namespace Microsoft::MSR::CNTK;
using namespace std;

typedef std::pair<std::wstring, std::vector<float>*> MapEntry;
typedef std::map<std::wstring, std::vector<float>*> Layer;

int main()
{
	IEvaluateModel<float> *model;
	GetEvalF(&model);
	const string s("deviceId=auto minibatchSize=1 modelPath=\"D:\\Models\\model.cmf\"");
	model->Init(s);
	std::vector<float> inputs(45);
	for (int i = 0; i < 45; i++)
		inputs[i] = 0.5;
	std::vector<float> outputs(3);
	Layer inputLayer;
	inputLayer.insert(MapEntry(L"features", &inputs));
	Layer outputLayer;
	outputLayer.insert(MapEntry(L"labels", &outputs));
	model->Evaluate(inputLayer, outputLayer);
	return 0;
}

