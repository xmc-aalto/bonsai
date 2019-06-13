// New header file: contains some helper functions for bonsai

#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <random>
#include <algorithm>

using namespace std;

bool isFloat(string myString);
vector<int> pick(int N, int k);
unordered_set<int> pickSet(int N, int k, mt19937& gen);
