// ForexEval.cpp: определяет экспортированные функции для приложения DLL.
//

#include "stdafx.h"

extern "C"  __declspec(dllexport) void __stdcall LoadModel(wchar_t* s);
extern "C" __declspec(dllexport) void __stdcall EvalModel(double* inp, double* out);

using namespace Microsoft::MSR::CNTK;

typedef std::pair<std::wstring, std::vector<double>*> MapEntry;
typedef std::map<std::wstring, std::vector<double>*> Layer;

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
	std::vector<double> v1(3);
	for (int i = 0; i < 45; i++)
		v[i] = inp[i];
	std::map<std::wstring, size_t> inDims;
	std::map<std::wstring, size_t> outDims;
	model->GetNodeDimensions(inDims, NodeGroup::nodeInput);
	model->GetNodeDimensions(outDims, NodeGroup::nodeOutput);
	auto inputLayerName = inDims.begin()->first;
	auto outputLayerName = outDims.begin()->first;
	Layer inputLayer;
	inputLayer.insert(MapEntry(inputLayerName, &v));
	Layer outputLayer;
	outputLayer.insert(MapEntry(outputLayerName, &v1));
	model->Evaluate(inputLayer, outputLayer);
	for (int i = 0; i < 3; i++)
		out[i] = v1[i];
}


