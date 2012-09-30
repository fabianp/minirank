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
// A subclass of SfWeightVector that computes cross-product features using a hashing
// function to project the features to a space of 2^hash_mask_bits_ features.  Collisions
// are summed.  Proof that this is a good idea is given in an ICML paper CITE LANGFORD.
//
// The end result is that this allows very fast computation of an approximate polynomial
// kernel of degree 2.

#ifndef SF_HASH_WEIGHT_VECTOR_H__
#define SF_HASH_WEIGHT_VECTOR_H__

#include "sf-hash-inline.h"
#include "sf-weight-vector.h"

using std::string;

class SfHashWeightVector : public SfWeightVector {
 public:
  // Construct a weight vector of dimenson 2^hash_mask_bits, with all weights initialized to
  // zero, scale_ set to 1, and squared_norm_ set to 0.  Also initializes the hash_mask
  // appropriately.
  SfHashWeightVector(int hash_mask_bits);

  // Constructs a weight vector from a string, using the defined number of hash_mask_bits.
  SfHashWeightVector(int hash_mask_bits,
		     const string& weight_vector_string);

  // Frees the array of weights.
  virtual ~SfHashWeightVector();

  // Computes inner product of <phi(x_scale * x), w>, where phi()
  // is a vector composed of all features in x and the cross-product
  // of all features in x, where each of these features is hashed
  // to some new feature id from 0 to 2^num_bits_for_hash_ - 1.
  virtual float InnerProduct(const SfSparseVector& x,
			     float x_scale = 1.0) const;
  
  // w += phi(x_scale * x), where phi is defined as for InnerProduct above. 
  virtual void AddVector(const SfSparseVector& x, float x_scale);

 private:
  // Disallowed.
  SfHashWeightVector();

  int hash_mask_bits_;
  int hash_mask_;
};

#endif  // SF_HASH_WEIGHT_VECTOR_H__
