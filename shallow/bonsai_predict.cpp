/*
  Code written by : Sujay Khandagale and Han Xiao 

  The code is based on the codebase written by Yashoteja Prabhu for Parabel published at WWW'18.
*/

#include <iostream>
#include <fstream>
#include <string>

#include "bonsai.h"

using namespace std;

#define PD_DBG if(false)

void help()
{
  cerr<<"Sample Usage :"<<endl;
  cerr<<"./bonsai_test [feature file name] [score file name] [model dir name] -T 1 -s 0 -t 3 -B 10 -d 0.98 -q 1 "<<endl<<endl;

  cerr<<"-T Number of threads to use. default=[value saved in trained model]"<< endl;
  cerr<<"-s Starting tree index. default=[value saved in trained model]"<< endl;
  cerr<<"-t Number of trees to be grown. default=[value saved in trained model]"<< endl;
  cerr<<"-B Beam search width. default=10]"<< endl;
  cerr<<"-d Disount value for parent svm score. default=[value saved in trained model (0.98)]"<< endl;	
  cerr<<"-q quiet option (0/1). default=[value saved in trained model]"<< endl;

  cerr<<"feature and score files are in sparse matrix format"<<endl;
  exit(1);
}

Param parse_param(int argc, char* argv[], string model_dir)
{
  Param param(model_dir+"/param");

  string opt;
  string sval;
  _float val;

  for(_int i=0; i<argc; i+=2)
    {
      opt = string(argv[i]);
      sval = string(argv[i+1]);
      val = stof(sval);

      if(opt=="-T")
	param.num_thread = (_int)val;
      else if(opt=="-s")
	param.start_tree = (_int)val;
      else if(opt=="-t")
	param.num_tree = (_int)val;
      else if(opt=="-B")
	param.beam_size = (_int)val;
      else if(opt=="-d")
	param.discount = (_float)val;
      else if(opt=="-q")
	param.quiet = (_bool)val;
    }

  return param;
}

int main(int argc, char* argv[])
{
  std::ios_base::sync_with_stdio(false);

  if(argc < 4)
    help();

  string ft_file = string(argv[1]);
  check_valid_filename(ft_file,true);
  SMatF* tst_X_Xf = new SMatF(ft_file);

  string score_file = string(argv[2]);
  check_valid_filename(score_file,false);

  string model_dir = string(argv[3]);
  check_valid_foldername(model_dir);

  Param param = parse_param(argc-4, argv+4, model_dir);

  _float prediction_time, model_size;
  SMatF* score_mat = predict_trees( tst_X_Xf, param, model_dir, prediction_time, model_size );

  cout << "prediction time: " << 1000*(prediction_time/tst_X_Xf->nc) << " ms" << endl; 
  cout << "model size: " << model_size/1e+6 << " mb" << endl; 

  PD_DBG { 
    cout << "score_mat.shape = " << score_mat->nr << "x" << score_mat->nc << endl; 
    for(_int i=0; i< score_mat->nc; i++){
      cout << "size at " << i << " = " << score_mat->size[i] << endl;
    }
  }
  score_mat->write(score_file);

  delete tst_X_Xf;
  delete score_mat;
}
