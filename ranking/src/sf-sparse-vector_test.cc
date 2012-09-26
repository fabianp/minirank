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
#include "sf-sparse-vector.h"

int main (int argc, char** argv) {
  // Test vector string with comment.
  char x1_string [100] = "1.0 qid:2 1:1.0 2:2.5 4:-2 #happy";
  SfSparseVector x1(x1_string);

  assert(x1.GetComment() == string("happy"));
  assert(x1.GetGroupId() == "2");
  assert(x1.GetY() == 1.0);
  assert(x1.ValueAt(1) == 1);
  assert(x1.FeatureAt(1) == 1);
  assert(x1.ValueAt(2) == 2.5);

  assert(x1.GetSquaredNorm() == 11.25);

  // Test vector string without comment.
  char x2_string [100] = "1.0 1:1.0 2:2.5 4:-2\n";
  SfSparseVector x2(x2_string);
  assert(x2.GetSquaredNorm() == 11.25);
  assert(x2.GetY() == 1.0);
  assert(x2.ValueAt(1) == 1);
  assert(x2.FeatureAt(1) == 1);
  assert(x2.ValueAt(2) == 2.5);
  assert(x2.GetGroupId() == "");

  // Test vector string with trailing space.
  char x3_string [100] = "1.0 1:1.0      2:2.5   4:-2 \n";
  SfSparseVector x3(x3_string);
  assert(x3.GetSquaredNorm() == 11.25);
  assert(x3.GetY() == 1.0);
  assert(x3.ValueAt(1) == 1);
  assert(x3.FeatureAt(1) == 1);
  assert(x3.ValueAt(2) == 2.5);

  SfSparseVector x4(x1, x2, 0.0);
  assert(x4.GetY() == 0.0);
  assert(x4.ValueAt(1) == 0.0);

  SfSparseVector x5(SfSparseVector("1.0 1:1.0 2:-1.0 3:4.5"),
		     SfSparseVector("0.0 2:2.0 4:3.0"),
		     -1.0);
  assert(x5.GetY() == -1.0);
  assert(x5.ValueAt(1) == 1.0 && x5.FeatureAt(1) == 1);
  assert(x5.ValueAt(2) == -3.0 && x5.FeatureAt(2) == 2);
  assert(x5.ValueAt(3) == 4.5 && x5.FeatureAt(3) == 3);
  assert(x5.ValueAt(4) == -3.0 && x5.FeatureAt(4) == 4);

  // Test vector string with group id.
  char x6_string [100] = "1.0 gid:3 1:1.0      2:2.5   4:-2 #note";
  SfSparseVector x6(x6_string);
  assert(x6.GetSquaredNorm() == 11.25);
  assert(x6.GetY() == 1.0);
  assert(x6.ValueAt(1) == 1);
  assert(x6.FeatureAt(1) == 1);
  assert(x6.ValueAt(2) == 2.5);
  assert(x6.GetGroupId() == "3");
  assert(x6.GetComment() == string("note"));

  // Test vector string with bias term.
  char x7_string [100] = "1.0 2:2.5 4:-2\n";
  SfSparseVector x7(x7_string, true);
  assert(x7.GetSquaredNorm() == 11.25);
  assert(x7.GetY() == 1.0);
  assert(x7.ValueAt(0) == 1);
  assert(x7.FeatureAt(0) == 0);
  assert(x7.ValueAt(2) == -2);
  assert(x7.FeatureAt(2) == 4);
  assert(x7.GetGroupId() == "");

  // Test vector string with no bias term.
  char x8_string [100] = "1.0 2:2.5 4:-2\n";
  SfSparseVector x8(x8_string, false);
  assert(x8.GetSquaredNorm() == 10.25);
  assert(x8.GetY() == 1.0);
  assert(x8.ValueAt(0) == 0);
  assert(x8.FeatureAt(0) == 0);
  assert(x8.ValueAt(2) == -2);
  assert(x8.FeatureAt(2) == 4);
  assert(x8.GetGroupId() == "");

  std::cout << argv[0] << ": PASS" << std::endl;
}
