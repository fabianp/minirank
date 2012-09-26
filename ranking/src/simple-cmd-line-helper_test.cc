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
// simple-cmd-line-helper_test.cc
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Unit test for simple-cmd-line-helper.h

#include "simple-cmd-line-helper.h"

#include <assert.h>
#include <iostream>

int main (int argc, char** argv) {
  string test_name = argv[0];

  AddFlag("--name", "name of person selling stuff.", string(""));
  AddFlag("--number", "number of people selling stuff.", int(0));
  AddFlag("--frac", "percentage of people selling stuff.", float(0.0));
  AddFlag("--ask", "should we ask people stuff?", bool(false));

  // Fill in dummy data for testing.
  char* cmdline[8] = { "./a.out", "--name", "no-name", "--number", "1",
			  "--frac", "0.5", "--ask"};
  argv = cmdline;
  argc = 8;

  ParseFlags(argc, argv);

  assert(CMD_LINE_STRINGS["--name"] == string("no-name"));
  assert(CMD_LINE_FLOATS["--frac"] == 0.5);
  assert(CMD_LINE_INTS["--number"] == 1);
  assert(CMD_LINE_BOOLS["--ask"] == true);

  std::cout << test_name << ": PASS" << std::endl;
}
