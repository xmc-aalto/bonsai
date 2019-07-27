/*
  Code written by : Sujay Khandagale and Han Xiao 

  The code is based on the codebase written by Yashoteja Prabhu for Parabel published at WWW'18.
*/

#include "bonsai.h"

using namespace std;

///////////////////// Modified_code_start /////////////////////

bool TR_DEBUG = false, PD_DEBUG = false, DEBUG_PARTITION_TO_ASSIGN_MAT = false;
bool KMEANS_DEBUG = false, KMPP_DEBUG = false;

_float EPSILON = 1e-5;

///////////////////// Modified_code_end /////////////////////

mutex mtx;
thread_local mt19937 reng; // random number generator used during training 

_int get_rand_num( _int siz )
{
  _llint r = reng();
  _int ans = r % siz;
  return ans;
}

Node* init_root( _int num_Y, _int max_depth )
{
  VecI lbls;
  for( _int i=0; i<num_Y; i++ )
    lbls.push_back(i);
  Node* root = new Node( lbls, 0, max_depth );
  return root;
}

void reset_d_with_s( pairIF* svec, _int siz, _float* dvec )
{
  // reset array to zero
  for( _int i=0; i<siz; i++ )
    dvec[ svec[i].first ] = 0;
}

void set_d_with_s( pairIF* svec, _int siz, _float* dvec )
{
  // copy array?
  for( _int i=0; i<siz; i++ )
    dvec[ svec[i].first ] = svec[i].second;
}

void init_2d_float( _int dim1, _int dim2, _float**& mat )
{
  mat = new _float*[ dim1 ];
  for( _int i=0; i<dim1; i++ )
    mat[i] = new _float[ dim2 ]; 
}

void delete_2d_float( _int dim1, _int dim2, _float**& mat )
{
  for( _int i=0; i<dim1; i++ )
    delete [] mat[i];
  delete [] mat;
  mat = NULL;
}

void reset_2d_float( _int dim1, _int dim2, _float**& mat )
{
  for( _int i=0; i<dim1; i++ )
    for( _int j=0; j<dim2; j++ )
      mat[i][j] = 0;
}

///////////////////// Modified_code_start /////////////////////

_float mult_s_s_vec( pairIF* v1, pairIF* v2, _int size1, _int size2 )
{
  // assume each list is sorted by index (`first` field)
  if(size1 == 0 || size2 == 0)
    return 0.0;

  _int i = 0, j = 0, ind1, ind2;
  _float prod = 0.0;
  ind1 = v1[i].first;
  ind2 = v2[j].first;

  while(true){
    if(ind1 == ind2) {
      // cout << "ind1 = ind2 = " << ind1 << endl;
      prod += (v1[i++].second * v2[j++].second);
      ind1 = v1[i].first;
      ind2 = v2[j].first;
    }

    while(ind1 < ind2) {
      if (i+1 >= size1) {
	i++;
	break;
      }
      ind1 = v1[++i].first;
    }

    if(i >= size1) break;

    while(ind1 > ind2) {
      if (j+1 >= size2) {
	j++;
	break;
      }
      ind2 = v2[++j].first;
    }

    if(j >= size2) break;
  }
  // cout << prod << endl;
  return prod;
}

///////////////////// Modified_code_end /////////////////////


_float mult_d_s_vec( _float* dvec, pairIF* svec, _int siz )
{
  // dot product of two vectors
  _float prod = 0;
  for( _int i=0; i<siz; i++ )
    {
      _int id = svec[i].first;
      _float val = svec[i].second;
      prod += dvec[ id ] * val;
    }
  return prod;
}

void add_s_to_d_vec( pairIF* svec, _int siz, _float* dvec )
{
  for( _int i=0; i<siz; i++ )
    {
      _int id = svec[i].first;
      _float val = svec[i].second;
      dvec[ id ] += val;
    }
}

_float get_norm_d_vec( _float* dvec, _int siz )
{
  _float norm = 0;
  for( _int i=0; i<siz; i++ )
    norm += SQ( dvec[i] );
  norm = sqrt( norm );
  return norm;
}

void div_d_vec_by_scalar( _float* dvec, _int siz, _float s )
{
  for( _int i=0; i<siz; i++)
    dvec[i] /= s;
}

void normalize_d_vec( _float* dvec, _int siz )
{
  _float norm = get_norm_d_vec( dvec, siz );
  if( norm>0 )
    div_d_vec_by_scalar( dvec, siz, norm );
}


#define GETI(i) (y[i]+1)
typedef signed char schar;

//void solve_l2r_lr_dual(const problem *prob, double *w, double eps, double Cp, double Cn)
void solve_l2r_lr_dual( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int svm_iter )
{
  _int l = X_Xf->nc;
  _int w_size = X_Xf->nr;
  _int i, s, iter = 0;

  _double *xTx = new _double[l];
  _int max_iter = svm_iter;
  _int *index = new _int[l];	
  _double *alpha = new _double[2*l]; // store alpha and C - alpha
  _int max_inner_iter = 100; // for inner Newton
  _double innereps = 1e-2;
  _double innereps_min = min(1e-8, (_double)eps);
  _double upper_bound[3] = {Cn, 0, Cp};

  _int* size = X_Xf->size;
  pairIF** data = X_Xf->data;

  // Initial alpha can be set here. Note that
  // 0 < alpha[i] < upper_bound[GETI(i)]
  // alpha[2*i] + alpha[2*i+1] = upper_bound[GETI(i)]
  for(i=0; i<l; i++)
    {
      alpha[2*i] = min(0.001*upper_bound[GETI(i)], 1e-8);
      alpha[2*i+1] = upper_bound[GETI(i)] - alpha[2*i];
    }

  for(i=0; i<w_size; i++)
    w[i] = 0;

  for(i=0; i<l; i++)
    {
      xTx[i] = sparse_operator::nrm2_sq( size[i], data[i] );
      sparse_operator::axpy(y[i]*alpha[2*i], size[i], data[i], w);
      index[i] = i;
    }

  while (iter < max_iter)
    {
      for (i=0; i<l; i++)
	{
	  _int j = i + get_rand_num( l-i );
	  swap(index[i], index[j]);
	}

      _int newton_iter = 0;
      _double Gmax = 0;
      for (s=0; s<l; s++)
	{
	  i = index[s];
	  const _int yi = y[i];
	  _double C = upper_bound[GETI(i)];
	  _double ywTx = 0, xisq = xTx[i];
	  ywTx = yi*sparse_operator::dot( w, size[i], data[i] );
	  _double a = xisq, b = ywTx;

	  // Decide to minimize g_1(z) or g_2(z)
	  _int ind1 = 2*i, ind2 = 2*i+1, sign = 1;
	  if(0.5*a*(alpha[ind2]-alpha[ind1])+b < 0)
	    {
	      ind1 = 2*i+1;
	      ind2 = 2*i;
	      sign = -1;
	    }

	  //  g_t(z) = z*log(z) + (C-z)*log(C-z) + 0.5a(z-alpha_old)^2 + sign*b(z-alpha_old)
	  _double alpha_old = alpha[ind1];
	  _double z = alpha_old;
	  if(C - z < 0.5 * C)
	    z = 0.1*z;
	  _double gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
	  Gmax = max(Gmax, fabs(gp));

	  // Newton method on the sub-problem
	  const _double eta = 0.1; // xi in the paper
	  _int inner_iter = 0;
	  while (inner_iter <= max_inner_iter)
	    {
	      if(fabs(gp) < innereps)
		break;
	      _double gpp = a + C/(C-z)/z;
	      _double tmpz = z - gp/gpp;
	      if(tmpz <= 0)
		z *= eta;
	      else // tmpz in (0, C)
		z = tmpz;
	      gp = a*(z-alpha_old)+sign*b+log(z/(C-z));
	      newton_iter++;
	      inner_iter++;
	    }

	  if(inner_iter > 0) // update w
	    {
	      alpha[ind1] = z;
	      alpha[ind2] = C-z;
	      sparse_operator::axpy(sign*(z-alpha_old)*yi, size[i], data[i], w);
	    }
	}

      iter++;

      /*
	if(iter % 10 == 0)
	info(".");
      */

      if(Gmax < eps)
	break;

      if(newton_iter <= l/10)
	innereps = max(innereps_min, 0.1*innereps);

    }

  /*
    info("\noptimization finished, #iter = %d\n",iter);
    if (iter >= max_iter)
    info("\nWARNING: reaching max number of iterations\nUsing -s 0 may be faster (also see FAQ)\n\n");

    // calculate objective value

    double v = 0;
    for(i=0; i<w_size; i++)
    v += w[i] * w[i];
    v *= 0.5;
    for(i=0; i<l; i++)
    v += alpha[2*i] * log(alpha[2*i]) + alpha[2*i+1] * log(alpha[2*i+1])
    - upper_bound[GETI(i)] * log(upper_bound[GETI(i)]);
    info("Objective value = %lf\n", v);
  */

  delete [] xTx;
  delete [] alpha;
  //delete [] y;
  delete [] index;
}

void solve_l2r_l1l2_svc( SMatF* X_Xf, _int* y, _float *w, _float eps, _float Cp, _float Cn, _int svm_iter )
{
  _int l = X_Xf->nc;
  _int w_size = X_Xf->nr;

  _int i, s, iter = 0;
  _float C, d, G;
  _float *QD = new _float[l];
  _int max_iter = svm_iter;
  _int *index = new _int[l];
  _float *alpha = new _float[l];
  _int active_size = l;

  _int tot_iter = 0;

  // PG: projected gradient, for shrinking and stopping
  _float PG;
  _float PGmax_old = INF;
  _float PGmin_old = -INF;
  _float PGmax_new, PGmin_new;

  // default solver_type: L2R_L2LOSS_SVC_DUAL
  _float diag[3] = {(_float)0.5/Cn, (_float)0, (_float)0.5/Cp};
  _float upper_bound[3] = {INF, 0, INF};

  _int* size = X_Xf->size;
  pairIF** data = X_Xf->data;

  //d = pwd;
  //Initial alpha can be set here. Note that
  // 0 <= alpha[i] <= upper_bound[GETI(i)]

  for(i=0; i<l; i++)
    alpha[i] = 0;

  for(i=0; i<w_size; i++)
    w[i] = 0;

  for(i=0; i<l; i++)
    {
      QD[i] = diag[GETI(i)];

      //feature_node * const xi = prob->x[i];
      QD[i] += sparse_operator::nrm2_sq( size[i], data[i] );
      sparse_operator::axpy(y[i]*alpha[i], size[i], data[i], w);

      index[i] = i;
    }

  while (iter < max_iter)
    {
      PGmax_new = -INF;
      PGmin_new = INF;

      for (i=0; i<active_size; i++)
	{
	  _int j = i + get_rand_num( active_size-i );
	  swap(index[i], index[j]);
	}

      for (s=0; s<active_size; s++)
	{
	  tot_iter ++;

	  i = index[s];
	  const _int yi = y[i];
	  //feature_node * const xi = prob->x[i];

	  G = yi*sparse_operator::dot( w, size[i], data[i] )-1;

	  C = upper_bound[GETI(i)];
	  G += alpha[i]*diag[GETI(i)];

	  PG = 0;
	  if (alpha[i] == 0)
	    {
	      if (G > PGmax_old)
		{
		  active_size--;
		  swap(index[s], index[active_size]);
		  s--;
		  continue;
		}
	      else if (G < 0)
		PG = G;
	    }
	  else if (alpha[i] == C)
	    {
	      if (G < PGmin_old)
		{
		  active_size--;
		  swap(index[s], index[active_size]);
		  s--;
		  continue;
		}
	      else if (G > 0)
		PG = G;
	    }
	  else
	    PG = G;

	  PGmax_new = max(PGmax_new, PG);
	  PGmin_new = min(PGmin_new, PG);

	  if(fabs(PG) > 1.0e-12)
	    {
	      _float alpha_old = alpha[i];
	      alpha[i] = min(max(alpha[i] - G/QD[i], (_float)0.0), C);
	      d = (alpha[i] - alpha_old)*yi;
	      sparse_operator::axpy(d, size[i], data[i], w);
	    }
	}

      iter++;

      if(PGmax_new - PGmin_new <= eps)
	{
	  if(active_size == l)
	    break;
	  else
	    {
	      active_size = l;
	      PGmax_old = INF;
	      PGmin_old = -INF;
	      continue;
	    }
	}
      PGmax_old = PGmax_new;
      PGmin_old = PGmin_new;
      if (PGmax_old <= 0)
	PGmax_old = INF;
      if (PGmin_old >= 0)
	PGmin_old = -INF;
    }

  // calculate objective value

  delete [] QD;
  delete [] alpha;
  delete [] index;
}

SMatF* svms( SMatF* trn_X_Xf, SMatF* trn_Y_X, Param& param )
{
  _float eps = 0.1;
  //_float Cp = param.log_loss_coeff;
  //_float Cn = param.log_loss_coeff;
  //_float f = (_float)param.num_trn / (_float)trn_X_Xf->nc;
  _float f = 1.0;
  _float Cp = param.log_loss_coeff * f;
  _float Cn = param.log_loss_coeff * f;
  _float th = param.svm_th;

  _int num_Y = trn_Y_X->nc;
  _int num_X = trn_X_Xf->nc;
  _int num_Xf = trn_X_Xf->nr;

  _int* y = new _int[ num_X ];
  fill( y, y+num_X, -1 );

  SMatF* w_mat = new SMatF( num_Xf, num_Y );
  _float* w = new _float[ num_Xf ];

  for( _int l=0; l<num_Y; l++ )
    {
      for( _int i=0; i < trn_Y_X->size[ l ]; i++ )
	y[ trn_Y_X->data[l][i].first ] = +1;

      if( param.septype == L2R_L2LOSS_SVC )
	solve_l2r_l1l2_svc( trn_X_Xf, y, w, eps, Cp, Cn, param.svm_iter );
      else if( param.septype == L2R_LR )
	solve_l2r_lr_dual( trn_X_Xf, y, w, eps, Cp, Cn, param.svm_iter );

      w_mat->data[ l ] = new pairIF[ num_Xf ]();
      _int siz = 0;
      for( _int f=0; f<num_Xf; f++ )
	{
	  if( fabs( w[f] ) > th )
	    w_mat->data[ l ][ siz++ ] = make_pair( f, w[f] );
	}
      Realloc( num_Xf, siz, w_mat->data[ l ] );
      w_mat->size[ l ] = siz;

      for( _int i=0; i < trn_Y_X->size[ l ]; i++ )
	y[ trn_Y_X->data[l][i].first ] = -1;
    }

  delete [] y;
  delete [] w;

  return w_mat;
}



void reindex_rows( SMatF* mat, _int nr, VecI& rows )
{
  mat->nr = nr;
  for( _int i=0; i<mat->nc; i++ )
    {
      for( _int j=0; j<mat->size[i]; j++ )
	mat->data[i][j].first = rows[ mat->data[i][j].first ];
    }
}

thread_local _bool* mask; // shared among threads?
void active_dims( SMatF* mat, VecI& insts, VecI& dims, _llint& nnz )
{
  nnz = 0;
  dims.clear(); // rows become empty
  _int num_trn = mat->nc;
  _int num_dim = mat->nr;
  _int* size = mat->size;
  pairIF** data = mat->data;

  for(_int i=0; i<insts.size(); i++)
    {
      _int inst = insts[i];
      for(_int j=0; j<size[inst]; j++)
	{
	  _int dim = data[inst][j].first;
	  if(!mask[dim]) // avoid being added mutiple types
	    dims.push_back(dim);
	  mask[dim] = true;
	  nnz++;
	}
    }

  sort(dims.begin(),dims.end());

  for(_int i=0; i<dims.size(); i++)
    {
      _int dim = dims[i];
      mask[ dim ] = false;
    }
}

///////////////////// Modified_code_start /////////////////////

SMatF* partition_to_assign_mat( SMatF* Y_X, VecI& partition)
{
  _int num_Y = Y_X->nc;
  _int num_X = Y_X->nr;  
  // _int num_partitions = unordered_set<_int> (partition.begin(), partition.end()).size();
  _int num_partitions = *max_element(partition.begin(), partition.end()) + 1;

  if(DEBUG_PARTITION_TO_ASSIGN_MAT) {
    cout << "num_partitions: " << num_partitions << endl;
    cout << "num_Y: " << num_Y << endl;
  }

  vector< vector<_int> > Y_by_part(num_partitions);
  for( _int i=0; i<num_Y; i++ ) {
    if(DEBUG_PARTITION_TO_ASSIGN_MAT)
      cout << "adding " << i << " to partition " << partition[i] << endl;
    Y_by_part[ partition[i] ].push_back(i);  
  }

  /*
  VecI pos_Y, neg_Y;
  for( _int i=0; i<num_Y; i++ )
    {
      if( partition[i]==1 )
	pos_Y.push_back( i );
      else
	neg_Y.push_back( i );
    }
  */
  vector<_llint> nnz_by_part(num_partitions);
  vector< vector<_int> > X_by_part(num_partitions);
  for(_int i=0; i<num_partitions; i++){
    active_dims( Y_X, Y_by_part[i], X_by_part[i], nnz_by_part[i] );
  }
  /*
  _int pos_nnz, neg_nnz;
  VecI pos_X, neg_X;
  active_dims( Y_X, pos_Y, pos_X, pos_nnz );
  active_dims( Y_X, neg_Y, neg_X, neg_nnz );
  */
  
  if(DEBUG_PARTITION_TO_ASSIGN_MAT)
    cout << "assign_mat.shape: " << num_X << "x" << num_partitions  << endl;

  SMatF* assign_mat = new SMatF( num_X, num_partitions);  // num_partitions  columns, one for each side
  _int* size = assign_mat->size;
  pairIF** data = assign_mat->data;
  for(_int i=0; i<num_partitions; i++){

    if(DEBUG_PARTITION_TO_ASSIGN_MAT)
      cout << "size[" << i << "]=" << X_by_part[i].size() << endl;

    size[i] = X_by_part[i].size();

    if(DEBUG_PARTITION_TO_ASSIGN_MAT)
       cout << "data[" << i << "] = someting" << endl;

    data[i] = new pairIF[ size[i] ];
  }
    /*
  size[0] = pos_X.size();
  size[1] = neg_X.size();
  data[0] = new pairIF[ pos_X.size() ];
  data[1] = new pairIF[ neg_X.size() ];
    */
  for(_int i=0; i<num_partitions; i++){    
    for( _int j=0; j<X_by_part[i].size(); j++){
      data[i][j] = make_pair( X_by_part[i][j], 1 );
    }
  }

  /*
  for( _int i=0; i<pos_X.size(); i++)
    data[0][i] = make_pair( pos_X[i], 1 );

  for( _int i=0; i<neg_X.size(); i++)
    data[1][i] = make_pair( neg_X[i], 1 );
  */

  return assign_mat;
}

///////////////////// Modified_code_end /////////////////////

void shrink_mat( SMatF* mat, VecI& cols, SMatF*& s_mat, VecI& rows )
{
  // take cols and rows of mat, save it in s_mat
  // function as numpy.array slicing
  _int nc = mat->nc;
  _int nr = mat->nr;
  _int s_nc = cols.size();

  _int* size = mat->size;
  pairIF** data = mat->data;

  _llint nnz;

  // `rows` iss empty vector
  // values will be added after calling `active_dims`
  active_dims( mat, cols, rows, nnz );
  _int* maps = new _int[ nr ];
  for( _int i=0; i<rows.size(); i++ )
    maps[ rows[i] ] = i;

  _int s_nr = rows.size();
  s_mat = new SMatF( s_nr, s_nc, nnz, true );

  _llint sumsize = 0;
  for( _int i=0; i<s_nc; i++)
    {
      _int col = cols[i];
      s_mat->size[i] = size[ col ];
      s_mat->data[i] = s_mat->cdata + sumsize;
      sumsize += size[ col ];
    }

  for( _int i=0; i<s_nc; i++ )
    {	
      _int col = cols[ i ]; // get column number
      for( _int j=0; j<size[ col ]; j++ )
	{
	  _int row = maps[ data[ col ][ j ].first ];
	  _float val = data[ col ][ j ].second;
	  s_mat->data[i][j] = make_pair( row, val );
	}
    }

  delete [] maps;
}

void shrink_data_matrices_with_cent_mat( SMatF* trn_X_Xf, SMatF* trn_Y_X, SMatF* cent_mat, VecI& n_Y, Param& param, SMatF*& n_trn_X_Xf, SMatF*& n_trn_Y_X, SMatF*& n_cent_mat, VecI& n_X, VecI& n_Xf, VecI& n_cXf )
{
  shrink_mat( trn_Y_X, n_Y, n_trn_Y_X, n_X );
  shrink_mat( trn_X_Xf, n_X, n_trn_X_Xf, n_Xf );
  shrink_mat( cent_mat, n_Y, n_cent_mat, n_cXf );
}

void shrink_data_matrices( SMatF* trn_X_Xf, SMatF* trn_Y_X, VecI& n_Y, Param& param, SMatF*& n_trn_X_Xf, SMatF*& n_trn_Y_X, VecI& n_X, VecI& n_Xf)
{
  shrink_mat( trn_Y_X, n_Y, n_trn_Y_X, n_X );
  shrink_mat( trn_X_Xf, n_X, n_trn_X_Xf, n_Xf );
  // shrink_mat( cent_mat, n_Y, n_cent_mat, n_cXf );
}

void train_leaf_svms( Node* node, SMatF* X_Xf, SMatF* Y_X, _int nr, VecI& n_Xf, Param& param )
{
  SMatF* w_mat = svms( X_Xf, Y_X, param );
  reindex_rows( w_mat, nr, n_Xf );
  node->w = w_mat;
}

///////////////////// Modified_code_start /////////////////////

void squeeze_partition_array(vector<_int> &partition){
  // make the elements in partition such that they're contiguous integers  
  unordered_set<_int> partition_set = unordered_set<_int> (partition.begin(), partition.end());
  vector<_int> unique_part_ids (partition_set.begin(), partition_set.end());
  vector<_int> old2new(partition.size());
  for(_int i=0; i<unique_part_ids.size(); i++)
    old2new[unique_part_ids[i]] = i;
  for(_int i=0; i<partition.size(); i++)
    partition[i] = old2new[partition[i]];
}

void kmeans( SMatF* mat, _float acc, VecI& partition, _int K) {
  // acc: accuracy parameter, when smaller than this, the iteration stops
  // should be epsilon
  Timer timer;
  timer.resume();

  _int nc = mat->nc;  // number of data points
  _int nr = mat->nr; // feature dim
  
  vector<_int> c(K);

    c = pick(nc, K);
  
  
  if(KMEANS_DEBUG) {
    cout << "c: ";
    std::copy(c.begin(), c.end(), std::ostream_iterator<int>(std::cout, " "));
    cout << endl;
  }

  _float** centers;
  init_2d_float(K, nr, centers );
  reset_2d_float(K, nr, centers );
  for( _int i=0; i<K; i++ )
    set_d_with_s( mat->data[c[i]], mat->size[c[i]], centers[i] );

  _float** cosines;
  init_2d_float( K, nc, cosines );
	
  partition.resize( nc );

  _float old_cos = -10000;
  _float new_cos = -1;
  _int best_center = 0;
  _float best_sim  = 0;

  while( new_cos - old_cos >= acc ) {

    for( _int i=0; i<K; i++ ) {
      for( _int j=0; j<nc; j++ )
	cosines[i][j] = mult_d_s_vec( centers[i], mat->data[j], mat->size[j] ); // compute cosine sim
    }

    if(KMEANS_DEBUG)
      cout << "cosine done" << endl;


    old_cos = new_cos;
    new_cos = 0;
    for( _int j=0; j<nc; j++ ) {
      best_sim = 0;
      best_center = 0;
      for( _int i=0; i<K; i++){
	// cout << "cosines[i][j]=" << cosines[i][j] << endl;;
	if(cosines[i][j] > best_sim){
	  best_sim = cosines[i][j];
	  best_center = i;
	}
      }
      assert(best_center >= 0);
      assert(best_center < K);

      partition[j] = best_center;
      new_cos += best_sim;
    }

    if(KMEANS_DEBUG)
      cout << "find closest neighbour done" << endl;

    new_cos /= nc;

    reset_2d_float( K, nr, centers );

    for( _int i=0; i<nc; i++ ) {
      _int p = partition[ i ];
      add_s_to_d_vec( mat->data[i], mat->size[i], centers[ p ] );
    }

    for( _int i=0; i<K; i++ )
      normalize_d_vec( centers[i], nr );

    if(KMEANS_DEBUG) {
      cout << "re-centering done" << endl;
      cout << "new_cos - old_cos = " << new_cos - old_cos << endl;
    }
  }
  

  // for partition with one element,
  // marge it to the closest partition

  // get label frequency first
  vector<_int> label_count(partition.size());
  for(auto l : partition)
    label_count[l] += 1;

  bool changed = false;
  for(_int j=0; j<partition.size(); j++){
    if(label_count[partition[j]] == 1){
      cout << "label " << partition[j] << " is singleton" << endl;
      // to cloest partition
      best_center = 0;
      best_sim = 0;
      for( _int i=0; i<K; i++){
	// not itself
	if(cosines[i][j] > best_sim && i != partition[j]){
	  best_sim = cosines[i][j];
	  best_center = i;
	}
      }
      assert(best_center >= 0);
      assert(best_center < K);

      cout << "gets assigned to partition " << best_center << endl;
      partition[j] = best_center;
      // now a new member for `best_center`
      label_count[best_center]++;
    }
  }

  // there can be empty clusters
  // reindex the partition s.t. partition ids are contiguous
  squeeze_partition_array(partition);
  // unordered_set<_int> partition_set = unordered_set<_int> (partition.begin(), partition.end());
  // vector<_int> unique_part_ids (partition_set.begin(), partition_set.end());
  // vector<_int> old2new(partition.size());
  // for(_int i=0; i<unique_part_ids.size(); i++)
  //   old2new[unique_part_ids[i]] = i;
  // for(_int i=0; i<partition.size(); i++)
  //   partition[i] = old2new[partition[i]];  

  
  // release memory
  delete_2d_float( K, nr, centers );
  delete_2d_float( K, nc, cosines );
  if(KMEANS_DEBUG)
    cout << "delete done" << endl;

  cout << "time spent on kmeans: " << timer.stop() << " secs" << endl;
}

///////////////////// Modified_code_end /////////////////////

void split_node_kmeans( Node* node, SMatF* X_Xf, SMatF* Y_X, SMatF* cent_mat, _int nr, VecI& n_Xf, VecI& partition, Param& param ) {
  if(KMEANS_DEBUG)
    cout << "partition using KMEANS" << endl;
  // change to graph partitioning
  kmeans( cent_mat, param.kmeans_eps, partition, param.num_children);

  // get the assignment matrix of 2 columns, whether left or right
  if(KMEANS_DEBUG){
    cout << "partition.size() = " << partition.size() << endl;
    cout << "Y_X.shape = " << Y_X->nr << "x" << Y_X->nc << endl;
  }
  SMatF* assign_mat = partition_to_assign_mat( Y_X, partition );

  if(KMEANS_DEBUG)
    cout << "partition_to_assign_mat done" << endl;

  SMatF* w_mat = svms( X_Xf, assign_mat, param );
  reindex_rows( w_mat, nr, n_Xf );
  node->w = w_mat;

  if(KMEANS_DEBUG)
    cout << "svm done" << endl;

  delete assign_mat;
}

///////////////////// Modified_code_start /////////////////////

Tree* train_tree( SMatF* trn_X_Xf, SMatF* trn_Y_X, SMatF* cent_mat, Param& param, _int tree_no )
{
  reng.seed(tree_no);
  _int num_X = trn_X_Xf->nc;
  _int num_Xf = trn_X_Xf->nr;
  _int num_Y = trn_Y_X->nc;
  _int num_XY = cent_mat->nr;

  // ---------------
	  
  _int max_n = max( max( max( num_X+1, num_Xf+1 ), num_Y+1 ), num_XY+1);
  mask = new _bool[ max_n ]();

  // TOOD: should be a parameter
  // _int max_depth = ceil( log2( num_Y/param.max_leaf ) ) + 1; 
  _int max_depth = param.max_depth;
	
  Tree* tree = new Tree;
  vector<Node*>& nodes = tree->nodes;

  Node* root = init_root(num_Y, max_depth);
		
  nodes.push_back( root );

  for(_int i=0; i<nodes.size(); i++) {
      if( i%100==0 )
	cout << "node " << i << endl;

      Node* node = nodes[i];

      if(TR_DEBUG)
	cout << "at node " << i << ", depth " << node->depth << ", is_leaf " << node->is_leaf << endl;

      VecI& n_Y = node->Y; // labels in node, vector of integer
      SMatF* n_trn_X_Xf = NULL; // feature matrix in node
      SMatF* n_trn_Y_X = NULL; // label matrix in node
      SMatF* n_cent_mat = NULL; // centroid matrix in node
      VecI n_X; 
      VecI n_Xf;
      VecI n_cXf;

      // slice the matrix by rows and columns

      shrink_data_matrices_with_cent_mat( trn_X_Xf, trn_Y_X, cent_mat, n_Y, param, n_trn_X_Xf, n_trn_Y_X, n_cent_mat, n_X, n_Xf, n_cXf );


      if( node->is_leaf ) {
	cout << "train leaf node " <<  i << " with " << node->Y.size() << " labels" << endl;
	train_leaf_svms( node, n_trn_X_Xf, n_trn_Y_X, num_Xf, n_Xf, param );
      }
      else
	{
	  // initialized to be empty
	  VecI partition; // partitioning starting from 0
	  cout << "split internal node" << endl;

	    split_node_kmeans( node, n_trn_X_Xf, n_trn_Y_X, n_cent_mat, num_Xf, n_Xf, partition, param );

	  
	  _int n_effective_partitions = unordered_set<_int>(partition.begin(), partition.end()).size();
	  
	  cout << "n_effective_partitions=" << n_effective_partitions << endl;

	  vector< vector<_int> > labels_by_child(n_effective_partitions);
	  for( _int j=0; j<n_Y.size(); j++){
	    assert(partition[j] >= 0);
	    assert(partition[j] < n_effective_partitions);
	    // cout << "partition[j]=" << partition[j] << endl;
	    // cout << "param.num_children=" << param.num_children << endl;	   
	    labels_by_child[ partition[j] ].push_back( n_Y[j] );
	  }

	  if(TR_DEBUG)
	    cout << "populating labels_by_child done" << endl;
	  /*
	  VecI pos_Y, neg_Y;
	  for( _int j=0; j<n_Y.size(); j++ )
	    if( partition[j] )
	      pos_Y.push_back( n_Y[ j ] );
	    else
	      neg_Y.push_back( n_Y[ j ] );
	  */

	  if(TR_DEBUG) {
	    cout << "label partition sizes: " << endl;
	    for(auto  child_labels: labels_by_child) 
	      cout << child_labels.size() << ", ";
	    cout << endl;
	    /*
	    for(vector<_int>  child_labels: labels_by_child) {
	      copy(child_labels.begin(), child_labels.end(), ostream_iterator<_int>(cout, " "));
	      cout << endl;
	    }
	    */
	  }

	  for(vector<_int>  child_labels: labels_by_child) {
	    Node* child_node = new Node( child_labels, node->depth+1, max_depth );

	    // when not enough labels to partition, make it a leaf  
	    if(child_labels.size() <= param.num_children)
	      child_node->is_leaf = true;

	    nodes.push_back( child_node );
	    node->children.push_back( nodes.size()-1 );
	  }
	  
	  if(TR_DEBUG)
	    cout << "nodes added to tree" << endl;
	  /*
	  Node* pos_node = new Node( pos_Y, node->depth+1, max_depth );
	  nodes.push_back( pos_node );
	  node->pos_child = nodes.size()-1;

	  Node* neg_node = new Node( neg_Y, node->depth+1, max_depth );
	  nodes.push_back(neg_node);
	  node->neg_child = nodes.size()-1;
	  */
	}

      delete n_trn_X_Xf;
      delete n_trn_Y_X;

      if(n_cent_mat != NULL) {
	if(TR_DEBUG)
	  cout << "deleting n_cent_mat" << endl;
	delete n_cent_mat;
      }

      if(TR_DEBUG)
	cout << "delete done" << endl;
    }
  tree->num_Xf = num_Xf;
  tree->num_Y = num_Y;

  delete [] mask;
	
  return tree;
}

///////////////////// Modified_code_end /////////////////////

void append_bias( SMatF* mat, _float bias )
{
  _int nc = mat->nc;
  _int nr = mat->nr;
  _int* size = mat->size;
  pairIF** data = mat->data;

  for( _int i=0; i<nc; i++ )
    {
      _int siz = size[i];
      Realloc( siz, siz+1, data[i] );
      data[i][siz] = make_pair( nr, bias );
      size[i]++;	
    }
  (mat->nr)++;
}

void train_trees_thread( SMatF* trn_X_Xf, SMatF* trn_Y_X, SMatF* cent_mat, Param param, _int s, _int t, string model_dir, _float* train_time )
{
  Timer timer;
	
  for(_int i=s; i<s+t; i++)
    {
      timer.resume();
      cout<<"tree "<<i<<" training started"<<endl;

      Tree* tree = train_tree( trn_X_Xf, trn_Y_X, cent_mat, param, i );
      timer.stop();

      tree->write( model_dir, i );

      timer.resume();
      delete tree;

      cout<<"tree "<<i<<" training completed"<<endl;
      timer.stop();
    }
  {
    timer.resume();
    lock_guard<mutex> lock(mtx);
    *train_time += timer.stop();
  }
}

void train_trees( SMatF* trn_X_Xf, SMatF* trn_X_Y, SMatF* trn_X_XY, Param& param, string model_dir, _float& train_time )
{
  // called by main
  // train trees in parallel
  _float* t_time = new _float;
  *t_time = 0;
  Timer timer;
	
  timer.start();
  param.num_trn = trn_X_Xf->nc;
  trn_X_Xf->unit_normalize_columns();
  SMatF* trn_Y_X = trn_X_Y->transpose(); // each column a training sample

  SMatF* cent_mat = NULL;

    // cent_mat = trn_X_Xf->prod( trn_Y_X ); // get the label matrix , each column a label
    // cent_mat->unit_normalize_columns();  
    // cent_mat->threshold( param.cent_th ); // make it sparse by thresholding
  
  if(param.cent_type == 0)
  {
    cent_mat = trn_X_Xf->prod( trn_Y_X );
    cent_mat->unit_normalize_columns();
  }

  else if(param.cent_type == 1)
  {
    cent_mat = trn_X_Y->prod( trn_Y_X ); 
    cent_mat->remove_self_coocc(0);  //passing 0 instead of param.num_Xf
    cent_mat->unit_normalize_columns();
  }

  else if(param.cent_type == 2)
  {
    cent_mat = trn_X_XY->prod( trn_Y_X ); // get the label matrix , each column a label
    // cent_mat->unit_normalize_columns(); 
    cent_mat->unit_normalize_X_columns(param.num_Xf, param.num_Y);  //changed
    // cent_mat->unit_normalize_Y_columns(param.num_Xf, param.num_Y);
    // cent_mat->normalize_Y_columns(param.num_Xf, param.num_Y);
    cent_mat->remove_self_coocc(param.num_Xf);
    cent_mat->unit_normalize_Y_columns(param.num_Xf, param.num_Y);
    // cent_mat->make_coooc_cons(param.num_Xf, 1);
  }

  cent_mat->threshold( param.cent_th ); // make it sparse by thresholding

  append_bias( trn_X_Xf, param.bias );

  _int tree_per_thread = (_int)ceil((_float)param.num_tree/param.num_thread);
  vector<thread> threads;
  _int s = param.start_tree; // the tree id?
  for( _int i=0; i<param.num_thread; i++ )
    {
      if( s < param.start_tree+param.num_tree )
	{
	  _int t = min( tree_per_thread, param.start_tree+param.num_tree-s );
	  threads.push_back( thread( train_trees_thread, trn_X_Xf, trn_Y_X, cent_mat, param, s, t, model_dir, ref(t_time) ));
	  s += t;
	}
    }
  timer.stop();	

  for(_int i=0; i<threads.size(); i++)
    threads[i].join();

  timer.resume();
  delete trn_Y_X;
  delete cent_mat;

  *t_time += timer.stop();
  train_time = *t_time;
  delete t_time;
}

thread_local float* densew;
void update_svm_scores( Node* node, SMatF* tst_X_Xf, SMatF* score_mat, SMatI* id_type_mat, _float discount, _Septype septype ) {
  SMatF* w_mat = node->w;

  // number of SVM classifiers?
  // 2 for internal nodes
  // num. labels for leaf nodes
  _int num_svm = w_mat->nc;

  for( _int i=0; i<num_svm; i++ ) {
    _int target; // the 'label', can be real label or child label
    _int id_type;
    if( node->is_leaf ) {
      target = node->Y[i];
      id_type = -1;
    }
    else {
      target = node->children[i];
      id_type = 1;
    }

    set_d_with_s( w_mat->data[i], w_mat->size[i], densew );

    VecIF& X = node->X;
    for( _int j=0; j<X.size(); j++ ) { // for each testing point      
	_int inst = X[j].first; // the data point id
	_float oldvalue = X[j].second; // where does old value come from ???
	_float prod = mult_d_s_vec( densew, tst_X_Xf->data[ inst ], tst_X_Xf->size[ inst ] ); // dot product
	_float newvalue;
	if( septype == L2R_L2LOSS_SVC )
	  newvalue = - SQ( max( (_float)0.0, 1-prod ) ); // squared hinge loss
	else if( septype == L2R_LR )
	  newvalue = - log( 1 + exp( -prod ) ); // logistic loss

	newvalue += discount * oldvalue; // ??? what's `discount`?

	// update the "child" or "label"'s score
	score_mat->data[ inst ][ score_mat->size[ inst ]++ ] = make_pair( target, newvalue );
	id_type_mat->data[ inst ][ id_type_mat->size[ inst ]++ ] = make_pair( target, id_type );
    }
    reset_d_with_s( w_mat->data[i], w_mat->size[i], densew );
  }
}

template <typename A, typename B>
void zip(
	 A *a,
	 B *b,
	 std::vector<std::pair<A,B>> &zipped,
	 _int length)
{
    for(size_t i=0; i<length; ++i)
    {
        zipped.push_back(std::make_pair(a[i], b[i]));
    }
}

template <typename A, typename B>
void unzip(
    const std::vector<std::pair<A, B>> &zipped, 
    A *a, 
    B *b,
    _int length)
{
    for(size_t i=0; i<length; i++)
    {
        a[i] = zipped[i].first;
        b[i] = zipped[i].second;
    }
}
void update_next_level( _int b, vector<Node*>& nodes, SMatF* score_mat, SMatI* id_type_mat, Param& param )
{
  // put value in score_mat->data to node->X
  
  // b: the node id
  // nodes: all nodes

  // only enters "if" if:
  // b is intermediatry node and 
  // b is the right-most/last node of the level
  if( !nodes[b]->is_leaf && b<nodes.size()-1 && nodes[b+1]->depth > nodes[b]->depth )
    {
      _int* size = score_mat->size;
      pairIF** data = score_mat->data;
      _int beam_size = param.beam_size;
      _int num_X = score_mat->nc; // num. cols = num. testing points

      for( _int i=0; i<num_X; i++ ) // each testing point
	{
	  // cout << "test point " << i << " / " << num_X << endl;

	  if( size[i] > beam_size )
	    {
	      // take the first `beam_size` nodes
	      // sort the nodes in the ith tesing point
	      // needs to sort id_type_mat using score_mat to reflect the new ordering
	      // sort( data[i], data[i]+size[i], comp_pair_by_second_desc<_int,_float> );
	      // size[i] = beam_size; // each testing point only records top `beam_size` scores/labels

	      vector<pair<pairIF, pairII>> zipped;
	      zip(score_mat->data[i], id_type_mat->data[i], zipped, score_mat->size[i]);

	      // Sort the vector of pairs
	      sort(zipped.begin(), zipped.end(), 
		   [&](pair<pairIF, pairII> a, pair<pairIF, pairII> b)
		   {
		     return a.first.second > b.first.second;
		   });

	      // Write the sorted pairs back to the original vectors
	      unzip(zipped, score_mat->data[i], id_type_mat->data[i], score_mat->size[i]);
	      score_mat->size[i] = beam_size; // each testing point only records top `beam_size` scores/labels
	    }
	  for( _int j=0; j<size[i]; j++ ) // for each child node/label
	    {
	      // need to check if jth target is a node or a label
	      if(id_type_mat->data[i][j].second == 1) {
		// j is a node, pass instance i to node j
		assert(data[i][j].first < nodes.size());
		Node* node = nodes[ data[i][j].first ];	      
		node->X.push_back( make_pair( i, data[i][j].second ) ); // put the score in		
	      }
	      	      

	    }
	  // does not release the memory
	  // but sets the "head" to begining
	  // so memory is reused
	  size[i] = 0;
	  id_type_mat->size[i] = 0;
	}
    }
}

void exponentiate_scores( SMatF* mat )
{
  // to softmax?
  _int nc = mat->nc;
  _int* size = mat->size;
  pairIF** data = mat->data;

  for( _int i=0; i<nc; i++ )
    {
      for( _int j=0; j<size[i]; j++)
	data[i][j].second = exp( data[i][j].second );

      sort( data[i], data[i]+size[i], comp_pair_by_first<_int,_float> );
    }
}

///////////////////// Modified_code_start /////////////////////

SMatF* predict_tree( SMatF* tst_X_Xf, Tree* tree, Param& param )
{
  _int num_X = tst_X_Xf->nc; //  number of test points
  _int num_Y = param.num_Y;  //  number of labels
  _int beam_size = param.beam_size;
  // _int max_leaf = param.max_leaf;

  // max_leaf deducted from the actual leaves
  // _int max_leaf = -1;
  vector<_int> leaf_sizes;
  for(Node* n: tree->nodes)
    if(n->is_leaf)
      // max_leaf = max(max_leaf, (_int) n->Y.size());
      leaf_sizes.push_back(n->Y.size());

  sort(leaf_sizes.rbegin(), leaf_sizes.rend()); // sort in DESC

  _int buffer_size = 0; // size of array to score label scores
  // sum up top k sizes of node labels
  for(_int i=0; i < beam_size; i++) {
    // cout << "buffer size: sum up " << leaf_sizes[i] << endl;
    buffer_size += leaf_sizes[i];
  }

    
  vector<Node*>& nodes = tree->nodes;
  _int num_node = nodes.size(); // number of nodes in the tree

  SMatF* score_mat = new SMatF( num_node, num_X ) ; // num. nodes X num. of testing points
  SMatI* id_type_mat = new SMatI( num_node, num_X ) ; // tracks the type of id, 0 for node and 1 for label
  
  for( _int i=0; i<num_X; i++ ) {
    score_mat->size[i] = 0;
    score_mat->data[i] = new pairIF[ buffer_size];

    id_type_mat->size[i] = 0;
    id_type_mat->data[i] = new pairII[ buffer_size];
  }

  if(PD_DEBUG)
    cout << "score_mat init done" << endl;

  Node* node = nodes[0]; // the root?

  if(PD_DEBUG)
    cout << "node[0] done" << endl;

  // X stores the instance ids to predict (for test data), (id, score)
  // what does score mean?
  node->X.clear(); 

  for( _int i=0; i<num_X; i++ )
    node->X.push_back( make_pair( i, 0) );

  if(PD_DEBUG)
    cout << "nodes.size(): " << nodes.size() << endl;

  for( _int i=0; i<nodes.size(); i++ ) {
    // predict at each node in tree
    // if( i%100==0 )
    Node* node = nodes[i];

    if(PD_DEBUG)
      cout << "at node " << i << ", depth " << node->depth << " #test instances "<< node->X.size() << " #labels " << node->Y.size() << ", is_leaf " << node->is_leaf << endl;    
    update_svm_scores( node, tst_X_Xf, score_mat, id_type_mat, param.discount, param.septype );

    if(PD_DEBUG)
      cout << "update_svm_scores done" << endl;

    // updated only once at each level
    update_next_level( i, nodes, score_mat, id_type_mat, param );
    if(PD_DEBUG)
      cout << "node " << i << " ended" << endl;
  }

  if(PD_DEBUG)
    cout << "exponentiate scores started "<< endl;  
  exponentiate_scores( score_mat );
  if(PD_DEBUG)
    cout << "exponentiate scores ended "<< endl;  
  score_mat->nr = num_Y;

  delete id_type_mat;  // release memory!
  return score_mat;
}

///////////////////// Modified_code_end /////////////////////

void predict_trees_thread( SMatF* tst_X_Xf, SMatF* score_mat, Param param, _int s, _int t, string model_dir, _float* prediction_time, _float* model_size ) {
  Timer timer;

  timer.start();
  densew = new _float[ tst_X_Xf->nr+1 ]();
  timer.stop();

  for(_int i=s; i<s+t; i++) {
    timer.resume();
    cout<<"tree "<<i<<" predicting started"<<endl;
    timer.stop();
    Tree* tree = new Tree( model_dir, i );

    timer.resume();
    SMatF* tree_score_mat = predict_tree( tst_X_Xf, tree, param );

    {
      lock_guard<mutex> lock(mtx);
      score_mat->add( tree_score_mat );
      *model_size += tree->get_ram();
    }

    delete tree;
    delete tree_score_mat;

    cout<<"tree "<<i<<" predicting completed"<<endl;
    timer.stop();
  }

  delete [] densew;

  {
    lock_guard<mutex> lock(mtx);
    *prediction_time += timer.stop();
  }
}

SMatF* predict_trees( SMatF* tst_X_Xf, Param& param, string model_dir, _float& prediction_time, _float& model_size ) {
  _float* p_time = new _float;
  *p_time = 0;

  _float* m_size = new _float;
  *m_size = 0;

  Timer timer;

  timer.start();
  tst_X_Xf->unit_normalize_columns();
  append_bias( tst_X_Xf, param.bias );

  _int num_X = tst_X_Xf->nc;
  _int num_Y = param.num_Y;

  SMatF* score_mat = new SMatF( num_Y, num_X );

  _int tree_per_thread = (_int)ceil( (_float)param.num_tree/param.num_thread );
  vector<thread> threads;

  _int s = param.start_tree;
  for( _int i=0; i<param.num_thread; i++ )
    {
      if( s < param.start_tree+param.num_tree )
	{
	  _int t = min(tree_per_thread, param.start_tree+param.num_tree-s);
	  threads.push_back( thread( predict_trees_thread, tst_X_Xf, ref(score_mat), param, s, t, model_dir, ref( p_time ), ref( m_size ) ));
	  s += t;
	}
    }
  timer.stop();
	
  for(_int i=0; i<threads.size(); i++)
    threads[i].join();

  timer.resume();
  for(_int i=0; i<score_mat->nc; i++)
    for(_int j=0; j<score_mat->size[i]; j++)
      score_mat->data[i][j].second /= param.num_tree;

  *p_time += timer.stop();
  prediction_time = *p_time;
  delete p_time;

  model_size = *m_size;
  delete m_size;

  for( _int i=0; i<score_mat->nc; i++ ) // for each testing point
    {
      _int siz = score_mat->size[i]; // number of labels with scores
      sort( score_mat->data[i], score_mat->data[i] + siz, comp_pair_by_second_desc<_int,_float> );
      _int newsiz = min( siz, 100 ); // report the top 100?
      Realloc( siz, newsiz, score_mat->data[i] );
      score_mat->size[i] = newsiz;
      sort( score_mat->data[i], score_mat->data[i] + newsiz, comp_pair_by_first<_int,_float> ); // sort by label id
    }

  return score_mat;
}
