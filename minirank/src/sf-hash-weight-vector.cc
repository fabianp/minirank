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
// sf-weight-vector.cc
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Implementation of sf-weight-vector.h

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "sf-hash-weight-vector.h"

//-------------------------------------------------------------------//
//---------------- SfHashWeightVector Public Methods ----------------//
//-------------------------------------------------------------------//

SfHashWeightVector::SfHashWeightVector(int hash_mask_bits) 
  : SfWeightVector(1 << hash_mask_bits),
    hash_mask_bits_(hash_mask_bits) {
  if (hash_mask_bits_ < 0) {
    std::cerr << "Illegal number of hash_mask_bits for of weight vector less than 1."
	      << std::endl << "hash_mask_bits__: " << dimensions_ << std::endl;
    exit(1);
  }
  hash_mask_ = SfHashMask(hash_mask_bits);

  std::cout << "hash_mask_ " << hash_mask_ << std::endl;
}

SfHashWeightVector::SfHashWeightVector(int hash_mask_bits,
				       const string& weight_vector_string) 
  : SfWeightVector(weight_vector_string),
    hash_mask_bits_(hash_mask_bits) {
  if (hash_mask_bits_ < 0) {
    std::cerr << "Illegal number of hash_mask_bits for of weight vector less than 1." << std::endl
	      << "hash_mask_bits__: " << dimensions_ << std::endl;
    exit(1);
  }
  hash_mask_ = SfHashMask(hash_mask_bits);
}

SfHashWeightVector::~SfHashWeightVector() {
  delete[] weights_;
}

float SfHashWeightVector::InnerProduct(const SfSparseVector& x,
				       float x_scale) const {
  float inner_product = 0.0;
  for (int i = 0; i < x.NumFeatures(); ++i) {
    inner_product +=
      weights_[SfHash(x.FeatureAt(i), hash_mask_)] * x.ValueAt(i);
  }
  for (int i = 0; i < x.NumFeatures(); ++i) {
    float x_i_value = x.ValueAt(i);
    int x_i_feature = x.FeatureAt(i);
    for (int j = i; j < x.NumFeatures(); ++j) {
      inner_product +=
	weights_[SfHash(x_i_feature, x.FeatureAt(j), hash_mask_)] * 
	x_i_value * x.ValueAt(j);
    }
  }
  inner_product *= x_scale;
  inner_product *= scale_;
  return inner_product;
}

void SfHashWeightVector::AddVector(const SfSparseVector& x,
				   float x_scale) {
  float inner_product = 0.0;
  float norm_x = 0.0;

  for (int i = 0; i < x.NumFeatures(); ++i) {
    float this_x_value = x.ValueAt(i) * x_scale;
    int this_x_feature = SfHash(x.FeatureAt(i), hash_mask_);
    if (this_x_feature >= dimensions_) {
      std::cerr << "Error: feature hash id " << this_x_feature
		<< " exceeds weight vector dimension " << dimensions_
		<< std::endl;
      exit(1);
    }
    norm_x += this_x_value * this_x_value;
    inner_product += weights_[this_x_feature] * this_x_value;
    weights_[this_x_feature] += this_x_value / scale_;
  }
  for (int i = 0; i < x.NumFeatures(); ++i) {
    float x_i_value = x.ValueAt(i);
    int x_i_feature = x.FeatureAt(i);
    for (int j = i; j < x.NumFeatures(); ++j) {
      float this_x_value = x_i_value * x.ValueAt(j) * x_scale;
      int this_x_feature = SfHash(x_i_feature, x.FeatureAt(j), hash_mask_);
      if (this_x_feature >= dimensions_) {
	std::cerr << "Error: cross-product feature hash id " << this_x_feature
		  << " exceeds weight vector dimension " << dimensions_;
	exit(1);
      }
      norm_x += this_x_value * this_x_value;
      inner_product += weights_[this_x_feature] * this_x_value;
      weights_[this_x_feature] += this_x_value / scale_;
    }
  }
  squared_norm_ += norm_x + (2.0 * scale_ * inner_product); 
}
