/*
  Code written by : Sujay Khandagale and Han Xiao 

  The code is based on the codebase written by Yashoteja Prabhu for Parabel published at WWW'18.
*/


#include <iostream>
#include <fstream>
#include <string>

#include "timer.h"
#include "bonsai.h"
#include "helpers.h"

using namespace std;

void help()
{
  cerr<<"Sample Usage :"<<endl;
  cerr<<"./bonsai_train [feature file name] [label file name] [model dir name] -T 1 -s 0 -t 3 -w 10 -b 1.0 -c 1.0 -m 3 -f 0.1 -fcent 0 -k 0.0001 -siter 20 -q 0"<<endl<<endl;

  cerr<<"-T Number of threads. default=1"<<endl;
  cerr<<"-s Starting index of the trees. default=0"<<endl;
  cerr<<"-t Number of trees to be grown. default=3"<<endl;
  cerr<<"-w Number of children for each node. default=10"<<endl;
  cerr<<"-b Feature bias value, extre feature value to be appended. default=1.0"<<endl;
  cerr<<"-c SVM weight co-efficient. default=1.0"<<endl;
  cerr<<"-m Maximum number of depths, default=2"<<endl;
  cerr<<"-f Svm weights threshold. default=0.1"<<endl;
  cerr<<"-fcent Centroid weights threshold. default=0"<<endl;
  cerr<<"-k Kmeans eps. default=0.0001"<<endl;
  cerr<<"-siter no. of svm iterations. default=20"<<endl;
  cerr<<"-q quiet option (0/1). default=0"<<endl;
  cerr<<"-stype linear separator type. 0=L2R_L2LOSS_SVC, 1=L2R_LR. default=L2R_L2LOSS_SVC"<<endl;
  cerr<<"-ctype centroid representation type. 0=Bonsai, 1=Label cooccurence, 2=Bonsai+Label cooccurence. default=0"<<endl;

  exit(1);
}

Param parse_param(_int argc, char* argv[])
{
  Param param;

  string opt;
  string sval;
  _float val;

  for(_int i=0; i<argc; i+=2)
    {
      opt = string(argv[i]);
      sval = string(argv[i+1]);

      if(isFloat(sval))
	val = stof(sval);

      if(opt=="-m")
	param.max_depth = (_int)val;
      else if(opt=="-b")
	param.bias = (_float)val;
      else if(opt=="-c")
	param.log_loss_coeff = (_float)val;
      else if(opt=="-T")
	param.num_thread = (_int)val;
      else if(opt=="-w")
	param.num_children = (_int)val;
      else if(opt=="-s")
	param.start_tree = (_int)val;
      else if(opt=="-t")
	param.num_tree = (_int)val;
      else if(opt=="-f")
	param.svm_th = (_float)val;
      else if(opt=="-fcent")
	param.cent_th = (_float)val;
      else if(opt=="-k")
	param.kmeans_eps = (_float)val;
      else if(opt=="-siter")
	param.svm_iter = (_int)val;
      else if(opt=="-q")
	param.quiet = (_bool)val;
      else if(opt=="-stype")
	param.septype = (_Septype)((_int)val);
      else if(opt=="-ptype")
	param.part_type = (_Parttype)((_int)val);
      else if(opt=="-ctype")
      param.cent_type = (_int)val;
    }

  return param;
}

int main(int argc, char* argv[])
{
  std::ios_base::sync_with_stdio(false);

  if(argc < 5)
    help();

  string ft_file = string( argv[1] );
  check_valid_filename( ft_file, true );
  SMatF* trn_X_Xf = new SMatF( ft_file );

  string lbl_file = string( argv[2] );
  check_valid_filename( lbl_file, true );
  SMatF* trn_X_Y = new SMatF(lbl_file);

  string ft_lbl_file = string( argv[3] );
  check_valid_filename( ft_lbl_file, true );
  SMatF* trn_X_XY = new SMatF(ft_lbl_file);

  string model_dir = string( argv[4] );
  check_valid_foldername( model_dir );

  Param param = parse_param( argc-5, argv+5 );
  param.num_Xf = trn_X_Xf->nr;
  param.num_Y = trn_X_Y->nr;
  param.write( model_dir+"/param" );

  _float train_time;
  train_trees( trn_X_Xf, trn_X_Y, trn_X_XY, param, model_dir, train_time );
  cout << "Training time: " << train_time << " s" << endl;

  delete trn_X_Xf;
  delete trn_X_Y;
}
