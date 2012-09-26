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
//
// Implementation of sf-sparse-vector.h

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "sf-sparse-vector.h"

//----------------------------------------------------------------//
//---------------- SfSparseVector Public Methods ----------------//
//----------------------------------------------------------------//
SfSparseVector::SfSparseVector(const char* in_string)
  : y_(0.0), 
    a_(0.0),
    squared_norm_(0.0),
    group_id_("") {
  NoBias();
  Init(in_string);
}

SfSparseVector::SfSparseVector(const char* in_string,
			       bool use_bias_term)
  : y_(0.0), 
    a_(0.0),
    squared_norm_(0.0),
    group_id_("") {
  if (use_bias_term) {
    SetBias();
  } else {
    NoBias();
  }
  Init(in_string);
}

SfSparseVector::SfSparseVector(const SfSparseVector& a,
				 const SfSparseVector& b,
				 float y) 
  : y_(y),
    a_(0.0),
    squared_norm_(0.0) {
  group_id_ = a.GetGroupId();
  int a_i = 0;
  int b_i = 0;
  while (a_i < a.NumFeatures() || b_i < b.NumFeatures()) {
    // a has no features remaining.
    if (!(a_i < a.NumFeatures())) {
      PushPair(b.FeatureAt(b_i), 0.0 - b.ValueAt(b_i));
      ++b_i;
      continue;
    }
    // b has no features remaining.
    if (!(b_i < b.NumFeatures())) {
      PushPair(a.FeatureAt(a_i), a.ValueAt(a_i));
      ++a_i;
      continue;
    }
    // a's current feature is less than b's current feature.
    if (b.FeatureAt(b_i) < a.FeatureAt(a_i)) {
      PushPair(b.FeatureAt(b_i), 0.0 - b.ValueAt(b_i));
      ++b_i;
      continue;
    }
    // b's current feature is less than a's current feature.
    if (a.FeatureAt(a_i) < b.FeatureAt(b_i)) {
      PushPair(a.FeatureAt(a_i), a.ValueAt(a_i));
      ++a_i;
      continue;
    }
    // a_i and b_i are pointing to the same feature.
    PushPair(a.FeatureAt(a_i), a.ValueAt(a_i) - b.ValueAt(b_i));
    ++a_i;
    ++b_i;
  }
}

string SfSparseVector::AsString() const {
  std::stringstream out_stream;
  out_stream << y_ << " ";
  for (int i = 0; i < NumFeatures(); ++i) {
    out_stream << FeatureAt(i) << ":" << ValueAt(i) << " ";
  }
  if (!comment_.empty()) {
    out_stream << "#" << comment_;
  }
  return out_stream.str();
}

void SfSparseVector::PushPair(int id, float value) {
  if (id > 0 && NumFeatures() > 0 && id <= FeatureAt(NumFeatures() - 1) ) {
    std::cerr << id << " vs. " << FeatureAt(NumFeatures() - 1) << std::endl;
    DieFormat("Features not in ascending sorted order.");
  }

  FeatureValuePair feature_value_pair;
  feature_value_pair.id_ = id;
  feature_value_pair.value_ = value;
  features_.push_back(feature_value_pair);
  squared_norm_ += value * value;
}

//----------------------------------------------------------------//
//--------------- SfSparseVector Private Methods ----------------//
//----------------------------------------------------------------//

void SfSparseVector::DieFormat(const string& reason) {
  std::cerr << "Wrong format for input data:\n  " << reason << std::endl;
  exit(1);
}

void SfSparseVector::Init(const char* in_string) {
  int length = strlen(in_string);
  if (length == 0) DieFormat("Empty example string.");
 
  // Get class label.
  if (!sscanf(in_string, "%f", &y_))
    DieFormat("Class label must be real number.");

  // Parse the group id, if any.
  const char* position;
  position = strchr(in_string, ' ') + 1;
  if ((position[0] >= 'a' && position[0] <= 'z') ||
      (position[0] >= 'A' && position[0] <= 'Z')) {
    position = strchr(position, ':') + 1;
    const char* end = strchr(position, ' ');
    char group_id_c_string[1000];
    strncpy(group_id_c_string, position, end - position);
    group_id_ = group_id_c_string;
    position = end + 1;
  } 

  // Get feature:value pairs.
  for ( ;
       (position < in_string + length 
	&& position - 1 != NULL
	&& position[0] != '#');
       position = strchr(position, ' ') + 1) {
    
    // Consume multiple spaces, if needed.
    if (position[0] == ' ' || position[0] == '\n' ||
	position[0] == '\v' || position[0] == '\r') {
      continue;
    };
    
    // Parse the feature-value pair.
    int id = atoi(position);
    position = strchr(position, ':') + 1;
    float value = atof(position);
    PushPair(id, value);
  }

  // Parse comment, if any.
  position = strchr(in_string, '#');
  if (position != NULL) {
    comment_ = string(position + 1);
  }
}
