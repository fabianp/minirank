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
// simple-cmd-line-helper.h
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Some simple helper functions for parsing commandline flags.
//
// Commandline flags are declared with the AddFlag method.
// ParseFlags(argc, argv) parses the commandline flags.
// 
// Flag values are stored in the following global maps:
//   CMD_LINE_BOOLS
//   CMD_LINE_FLOATS
//   CMD_LINE_INTS
//   CMD_LINE_STRINGS
//
// Descriptions of each flag are stored in the maps:
//   CMD_LINE_DESCRIPTIONS
//
// The ShowHelp() method prints a description of each commandline flag, and exits. 
//
// Example useage:
/*
#include "simple-cmd-line-helper.h"

#include <cstdlib>
#include <iostream>

int main (int argc, char** argv) {

  AddFlag("--name", "name of person selling stuff.", string(""));
  AddFlag("--number", "number of people selling stuff.", int(0));
  AddFlag("--frac", "percentage of people selling stuff.", float(0.0));
  AddFlag("--ask", "should we ask people stuff?", bool(false));

  ParseFlags(argc, argv);

  std::cout << "--frac" << " " << CMD_LINE_FLOATS["--frac"] << std::endl;
  std::cout << "--name" << " " << CMD_LINE_STRINGS["--name"] << std::endl;
  std::cout << "--number" << " " << CMD_LINE_INTS["--number"] << std::endl;
  std::cout << "--ask" << " " << CMD_LINE_BOOLS["--ask"] << std::endl;
}
*/

#ifndef SIMPLE_CMD_LINE_HELPER_H__
#define SIMPLE_CMD_LINE_HELPER_H__

#include <iostream>
#include <map>
#include <sstream>
#include <string>

using std::map;
using std::string;

// Global maps containing commandline flags.
map<string, bool> CMD_LINE_BOOLS;
map<string, float> CMD_LINE_FLOATS;
map<string, int> CMD_LINE_INTS;
map<string, string> CMD_LINE_STRINGS;
map<string, string> CMD_LINE_DESCRIPTIONS;

void AddFlag(const string& flag_name,
	     const string& description,
	     bool default_value) {
  if (CMD_LINE_DESCRIPTIONS.find(flag_name) != CMD_LINE_DESCRIPTIONS.end()) {
    std::cerr << "Error. " << flag_name << " appears more than once." << std::endl;
    exit(1);
  }
  CMD_LINE_DESCRIPTIONS[flag_name] = description;
  CMD_LINE_BOOLS[flag_name] = default_value;
}

void AddFlag(const string& flag_name,
	     const string& description,
	     float default_value) {
  if (CMD_LINE_DESCRIPTIONS.find(flag_name) != CMD_LINE_DESCRIPTIONS.end()) {
    std::cerr << "Error. " << flag_name << " appears more than once." << std::endl;
    exit(1);
  }
  CMD_LINE_DESCRIPTIONS[flag_name] = description;
  CMD_LINE_FLOATS[flag_name] = default_value;
}

void AddFlag(const string& flag_name,
	     const string& description,
	     int default_value) {
  if (CMD_LINE_DESCRIPTIONS.find(flag_name) != CMD_LINE_DESCRIPTIONS.end()) {
    std::cerr << "Error. " << flag_name << " appears more than once." << std::endl;
    exit(1);
  }
  CMD_LINE_DESCRIPTIONS[flag_name] = description;
  CMD_LINE_INTS[flag_name] = default_value;
}

void AddFlag(const string& flag_name,
	     const string& description,
	     string default_value) {
  if (CMD_LINE_DESCRIPTIONS.find(flag_name) != CMD_LINE_DESCRIPTIONS.end()) {
    std::cerr << "Error. " << flag_name << " appears more than once." << std::endl;
    exit(1);
  }
  CMD_LINE_DESCRIPTIONS[flag_name] = description;
  CMD_LINE_STRINGS[flag_name] = default_value;
}

void ShowHelp() {
  std::cout << "Command line flag options: " << std::endl;
  for (map<string,string>::iterator iter = CMD_LINE_DESCRIPTIONS.begin();
       iter != CMD_LINE_DESCRIPTIONS.end();
       iter++) {
    fprintf(stderr, "      %-20s", iter->first.c_str());
    fprintf(stderr, "  %s\n\n", iter->second.c_str());
  }
  std::cout << std::endl;
  exit(0);
}

bool ParseBoolFlag(char** argv, int* i) {
  if (CMD_LINE_BOOLS.find(argv[*i]) != CMD_LINE_BOOLS.end()) {
    CMD_LINE_BOOLS[argv[*i]] = true;
    ++(*i);
    return true;
  }
  return false;
}

bool ParseGeneralFlag(int argc,
		      char** argv,
		      int* i) {
  if (CMD_LINE_FLOATS.find(argv[*i]) != CMD_LINE_FLOATS.end() ||
      CMD_LINE_INTS.find(argv[*i]) != CMD_LINE_INTS.end() ||      
      CMD_LINE_STRINGS.find(argv[*i]) != CMD_LINE_STRINGS.end()) {
    if (*i + 1 >= argc || (argv[*i + 1])[0] == '-') {
      std::cerr << "Error.  " << argv[*i] << " needs a value, but is given none."
		<< std::endl;
      exit(1);
    }
    std::stringstream arg_stream(argv[(*i + 1)]);
        
    if (CMD_LINE_FLOATS.find(argv[*i]) != CMD_LINE_FLOATS.end()) {
      float value;
      arg_stream >> value;
      CMD_LINE_FLOATS[argv[*i]] = value;
      *i += 2;
      return true;
    }

    if (CMD_LINE_INTS.find(argv[*i]) != CMD_LINE_INTS.end()) {
      int value;
      arg_stream >> value;
      CMD_LINE_INTS[argv[*i]] = value;
      *i += 2;
      return true;
    }

    if (CMD_LINE_STRINGS.find(argv[*i]) != CMD_LINE_STRINGS.end()) {
      string value;
      arg_stream >> value;
      CMD_LINE_STRINGS[argv[*i]] = value;
      *i += 2;
      return true;
    }
  }    
  return false;
}

void ParseFlags(int argc, char** argv) {
  if (argc == 1) ShowHelp();

  int i = 1;
  while (i < argc) {
    bool good_parse = false;
    good_parse = good_parse || ParseBoolFlag(argv, &i);
    good_parse = good_parse || ParseGeneralFlag(argc, argv, &i);
    if (!good_parse) {
      std::cerr << "Error. " << argv[i] << " is not a valid flag." << std::endl;
      exit(1);	
    }
  }
}

#endif  // SIMPLE_CMD_LINE_HELPER_H__
