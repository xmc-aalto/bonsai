// New cpp file: contains some helper functions for bonsai

#include "helpers.h"

bool isFloat( string myString ) {
  istringstream iss(myString);
  float f;
  iss >> noskipws >> f; // noskipws considers leading whitespace invalid
  // Check the entire string was consumed and if either failbit or badbit is set
  return iss.eof() && !iss.fail();
  
}


vector<int> pick(int N, int k) {
  random_device rd;
  mt19937 gen(rd());

  unordered_set<int> elems = pickSet(N, k, gen);

  // ok, now we have a set of k elements. but now
  // it's in a [unknown] deterministic order.
  // so we have to shuffle it:

  vector<int> result(elems.begin(), elems.end());
  shuffle(result.begin(), result.end(), gen);
  return result;
}

unordered_set<int> pickSet(int N, int k, mt19937& gen)
{
  uniform_int_distribution<> dis(0, N-1);
  unordered_set<int> elems;

  while (elems.size() < k) {
    elems.insert(dis(gen));
  }

  return elems;
}

