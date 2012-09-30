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
// sf-weight-vector.h
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// A simple weight vector class used for training and classification
// with SfSaprseVector instances.  The dimensionality d of the data
// set must be specified at construction time, as the internal
// data is kept in a contiguous array for data concurrency.
//
// Note that the weight vector is composed of feature value pairs,
// and a scaling parameter scale_.  This scaling parameter allows for
// fast L2-norm regularization.
//
// The squared_norm_ member maintains the squared norm on all updates.

#ifndef SF_WEIGHT_VECTOR_H__
#define SF_WEIGHT_VECTOR_H__

#include "sf-sparse-vector.h"

using std::string;

class SfWeightVector {
 public:
  // Construct a weight vector of dimenson d, with all weights initialized to
  // zero, scale_ set to 1, and squared_norm_ set to 0.
  SfWeightVector(int dimensionality);

  // Constructs a weight vector from a string, which is identical in format
  // to that produced by the AsString() member method.
  SfWeightVector(const string& weight_vector_string);

  // Simple copy constructor, needed to allocate a new array of weights.
  SfWeightVector(const SfWeightVector& weight_vector);

  // Frees the array of weights.
  virtual ~SfWeightVector();

  // Re-scales weight vector to scale of 1, and then outputs each weight in
  // order, space separated.
  string AsString();

  // Computes inner product of <x_scale * x, w>
  virtual float InnerProduct(const SfSparseVector& x,
			     float x_scale = 1.0) const;

  // Computes inner product of <x_scale * (a - b), w>
  float InnerProductOnDifference(const SfSparseVector& a,
				 const SfSparseVector& b,
				 float x_scale = 1.0) const;

  // w += x_scale * x
  virtual void AddVector(const SfSparseVector& x, float x_scale);

  // w *= scaling_factor
  void ScaleBy(double scaling_factor);

  // Returns value of element w_index, taking internal scaling into account.
  float ValueOf(int index) const;

  // Project this vector into the L1 ball of radius lambda.
  void ProjectToL1Ball(float lambda);

  // Project this vector into the L1 ball of radius at most lambda, plus or
  // minus epsilon / 2.
  void ProjectToL1Ball(float lambda, float epsilon);
  
  // Getters.
  double GetSquaredNorm() const { return squared_norm_; }
  int GetDimensions() const { return dimensions_; }

 protected:
  void ScaleToOne();

  float* weights_;
  double scale_;
  double squared_norm_;
  int dimensions_;

 private:
  // Disallowed.
  SfWeightVector();
};

#endif
