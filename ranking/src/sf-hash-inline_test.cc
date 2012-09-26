//================================================================================//
// Copyright 2009 Google Inc.                                                     //
//                                                                                // 
// Licensed under the Apache License, Version 2.0 (the "License");                //
// you may not use this file except in compliance with the License.               //
// You may obtain a copy of the License at                                        //
//                                                                                //
//      http://www.apache.org/licenses/LICENSE-2.0                                //
//                                                                                //
// Unless required by applicable law or agreed to in writing, software            //
// distributed under the License is distributed on an "AS IS" BASIS,              //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.       //
// See the License for the specific language governing permissions and            //
// limitations under the License.                                                 //
//================================================================================//
//
#include <assert.h>
#include <iostream>
#include <vector>
#include "sf-hash-inline.h"

using namespace std;

int main (int argc, char** argv) {
  // Test the mask.
  int mask4 = SfHashMask(4);
  assert(mask4 == 15);

  int mask22 = SfHashMask(22);

  // Test single int hash.
  assert(SfHash(66, mask22) == 3436118);

  // Test two-int hash.
  assert(SfHash(87, 71, mask22) == 4111611);

  // Test hash of vector of ints.
  vector<int> v_ints;
  for (int i = 1; i < 10; ++i) {
    v_ints.push_back(i);
  }
  assert(SfHash(v_ints, mask22) == 3190187);

  std::cout << argv[0] << ": PASS" << std::endl;
}
