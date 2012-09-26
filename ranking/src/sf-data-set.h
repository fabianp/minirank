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
// sf-data-set.h
//
// Author: D. Sculley, December 2008
// dsculley@google.com or dsculley@cs.tufts.edu
//
// A data set object based on SfSparseVector, as defined in
// sf-sparse-vector.h.  Methods are provided for reading
// a data set into memory from a file, and accessing individual
// vectors within the data set.

#ifndef SF_DATA_SET_H__
#define SF_DATA_SET_H__

#include <string>
#include <vector>

#include "sf-sparse-vector.h"

class SfDataSet {
 public:
  // Empty data set.
  SfDataSet(bool use_bias_term);

  // Construct and fill a SfDataSet with data from the given file.
  // Use buffer_mb megabytes for the buffer.
  SfDataSet(const string& file_name, int buffer_mb, bool use_bias_term);

  // Debug string.
  string AsString() const;
  
  // Number of total examples in data set.
  long int NumExamples() const { return vectors_.size(); }

  // Returns a reference to the specified vector.
  const SfSparseVector& VectorAt (long int index) const;

  // Adds the vector represented by this svm-light format string
  // to the data set.
  void AddVector(const string& vector_string);
  void AddVector(const char* vector_string);
  // Adds a copy of the given vector, using label y.
  void AddLabeledVector(const SfSparseVector& x, float y);

 private:
  // Member containing all vectors in data set.
  vector<SfSparseVector> vectors_;
  // Should we add a bias term to each new vector in the data set?
  bool use_bias_term_;
};

#endif  // SF_DATA_SET_H__
