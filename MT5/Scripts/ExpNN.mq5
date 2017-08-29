//+------------------------------------------------------------------+
//|                                                        ExpNN.mq5 |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property script_show_inputs
//--- input parameters
input int      ExtDepth=7;
input int      ExtDeviation=5;
input int      ExtBackstep=3;
input int      NBars=50000;
input double   Split=0.95;
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart()
  {
//---
    int limit=Bars(Symbol(),Period());
    if(limit>NBars) limit=NBars;
    int h=iCustom(Symbol(),Period(),"Examples\\ZigZag",ExtDepth,ExtDeviation,ExtBackstep);
    double zh[],zl[];
    ArrayResize(zh,limit);
    ArrayResize(zl,limit);
    CopyBuffer(h,1,0,limit,zh);
    CopyBuffer(h,2,0,limit,zl);
    MqlRates his[];
    ArrayResize(his,limit+16);
    CopyRates(Symbol(),Period(),0,limit,his);
    int tr=(int)(limit*Split);
    int ot=FileOpen("train.txt",FILE_WRITE|FILE_UNICODE);
    for(int i=15;i<tr;i++)
    {
      string d="|Input ";
      for(int j=0;j<15;j++)
      {
         double delta=(his[i-j].high-his[i-j].low)/Point();
         double delta1=(his[i-j].high-his[i-j-1].high)/Point();
         double delta2=(his[i-j].low-his[i-j-1].low)/Point();
         d=d+DoubleToString(delta,1)+" "+DoubleToString(delta1,1)+" "+DoubleToString(delta2,1)+" ";
      }
      string d1="|Label 0.0 0.0 1.0";
      if(zh[i]>0) d1="|Label 1.0 0.0 0.0";
      if(zl[i]>0) d1="|Label 0.0 1.0 0.0";
      d=d+d1;
      FileWrite(ot,d);
    }
    FileClose(ot);
    ot=FileOpen("test.txt",FILE_WRITE|FILE_UNICODE);
    for(int i=tr;i<limit;i++)
    {
      string d="|Input ";
      for(int j=0;j<15;j++)
      {
         double delta=(his[i-j].high-his[i-j].low)/Point();
         double delta1=(his[i-j].high-his[i-j-1].high)/Point();
         double delta2=(his[i-j].low-his[i-j-1].low)/Point();
         d=d+DoubleToString(delta,1)+" "+DoubleToString(delta1,1)+" "+DoubleToString(delta2,1)+" ";
      }
      string d1="|Label 0.0 0.0 1.0";
      if(zh[i]>0) d1="|Label 1.0 0.0 0.0";
      if(zl[i]>0) d1="|Label 0.0 1.0 0.0";
      d=d+d1;
      FileWrite(ot,d);
    }
    FileClose(ot);
  }
//+------------------------------------------------------------------+
