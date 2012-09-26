//==========================================================================//
// Copyright 2009 Google Inc.                                               //
//                                                                          // 
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//      http://www.apache.org/licenses/LICENSE-2.0                          //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
//==========================================================================//
//
// sf-weight-vector.cc
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Implementation of sf-weight-vector.h

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "sf-weight-vector.h"

//----------------------------------------------------------------//
//---------------- SfWeightVector Public Methods ----------------//
//----------------------------------------------------------------//

SfWeightVector::SfWeightVector(int dimensionality) 
  : scale_(1.0),
    squared_norm_(0.0),
    dimensions_(dimensionality) {
  if (dimensions_ <= 0) {
    std::cerr << "Illegal dimensionality of weight vector less than 1."
	      << std::endl
	      << "dimensions_: " << dimensions_ << std::endl;
    exit(1);
  }

  weights_ = new float[dimensions_];
  if (weights_ == NULL) {
    std::cerr << "Not enough memory for weight vector of dimension: " 
	      <<  dimensions_ << std::endl;
    exit(1);
  }
  for (int i = 0; i < dimensions_; ++i) {
    weights_[i] = 0;
  }
}

SfWeightVector::SfWeightVector(const string& weight_vector_string) 
  : scale_(1.0),
    squared_norm_(0.0),
    dimensions_(0) {
  // Count dimensions in string.
  std::stringstream count_stream(weight_vector_string);
  float weight;
  while (count_stream >> weight) { 
    ++dimensions_;
  }
    
  // Allocate weights_.
  weights_ = new float[dimensions_];
  if (weights_ == NULL) {
    std::cerr << "Not enough memory for weight vector of dimension: " 
	      <<  dimensions_ << std::endl;
    exit(1);
  }
  
  // Fill weights_ from weights in string.
  std::stringstream weight_stream(weight_vector_string);
  for (int i = 0; i < dimensions_; ++i) {
    weight_stream >> weights_[i];
    squared_norm_ += weights_[i] * weights_[i];
  }
}

SfWeightVector::SfWeightVector(const SfWeightVector& weight_vector) {
  scale_ = weight_vector.scale_;
  squared_norm_ = weight_vector.squared_norm_;
  dimensions_ = weight_vector.dimensions_;

  weights_ = new float[dimensions_];
  if (weights_ == NULL) {
    std::cerr << "Not enough memory for weight vector of dimension: " 
	      <<  dimensions_ << std::endl;
    exit(1);
  }

  for (int i = 0; i < dimensions_; ++i) {
    weights_[i] = weight_vector.weights_[i];
  }
}

SfWeightVector::~SfWeightVector() {
  delete[] weights_;
}

string SfWeightVector::AsString() {
  ScaleToOne();
  std::stringstream out_string_stream;
  for (int i = 0; i < dimensions_; ++i) {
    out_string_stream << weights_[i];
    if (i < (dimensions_ - 1)) {
      out_string_stream << " ";
    }
  }
  return out_string_stream.str();
}

float SfWeightVector::InnerProduct(const SfSparseVector& x,
				    float x_scale) const {
  float inner_product = 0.0;
  for (int i = 0; i < x.NumFeatures(); ++i) {
    inner_product += weights_[x.FeatureAt(i)] * x.ValueAt(i);
  }
  inner_product *= x_scale;
  inner_product *= scale_;
  return inner_product;
}

float SfWeightVector::InnerProductOnDifference(const SfSparseVector& a,
					       const SfSparseVector& b,
					       float x_scale) const {
  //   <x_scale * (a - b), w>
  // = <x_scale * a - x_scale * b, w>
  // = <x_scale * a, w> + <-1.0 * x_scale * b, w>

  float inner_product = 0.0;
  inner_product += InnerProduct(a, x_scale);
  inner_product += InnerProduct(b, -1.0 * x_scale);
  return inner_product;
}

void SfWeightVector::AddVector(const SfSparseVector& x, float x_scale) {
  if (x.FeatureAt(x.NumFeatures() - 1) > dimensions_) {
    std::cerr << "Feature " << x.FeatureAt(x.NumFeatures() - 1) 
	      << " exceeds dimensionality of weight vector: " 
	      << dimensions_ << std::endl;
    std::cerr << x.AsString() << std::endl;
    exit(1);
  }

  float inner_product = 0.0;
  for (int i = 0; i < x.NumFeatures(); ++i) {
    float this_x_value = x.ValueAt(i) * x_scale;
    int this_x_feature = x.FeatureAt(i);
    inner_product += weights_[this_x_feature] * this_x_value;
    weights_[this_x_feature] += this_x_value / scale_;
  }
  squared_norm_ += x.GetSquaredNorm() * x_scale * x_scale +
    (2.0 * scale_ * inner_product); 
}

void SfWeightVector::ScaleBy(double scaling_factor) {
  // Take care of any numerical difficulties.
  if (scale_ < 0.00000000001) ScaleToOne();

  // Do the scaling.
  squared_norm_ *= (scaling_factor * scaling_factor);
  if (scaling_factor > 0.0) {
    scale_ *= scaling_factor;
  } else {
    std::cerr << "Error: scaling weight vector by non-positive value!\n " 
	      << "This can cause numerical errors in PEGASOS projection.\n "
	      << "This is likely due to too large a value of eta * lambda.\n "
	      << std::endl;
    exit(1);
  }
}

float SfWeightVector::ValueOf(int index) const {
  if (index < 0) {
    std::cerr << "Illegal index " << index << " in ValueOf. " << std::endl;
    exit(1);
  }
  if (index >= dimensions_) {
    return 0;
  }
  return weights_[index] * scale_;
}

void SfWeightVector::ProjectToL1Ball(float lambda, float epsilon) {
  // Re-scale lambda.
  lambda = lambda / scale_;

  // Bail out early if possible.
  float current_l1 = 0.0;
  float max_value = 0.0;
  vector<float> non_zeros;
  for (int i = 0; i < dimensions_; ++i) {
    if (weights_[i] != 0.0) {
      non_zeros.push_back(fabsf(weights_[i]));
    } else {
      continue;
    }
    current_l1 += fabsf(weights_[i]);
    if (fabs(weights_[i]) > max_value) {
      max_value = fabs(weights_[i]);
    }
  }
  if (current_l1 <= (1.0 + epsilon) * lambda) return;

  float min = 0;
  float max = max_value;
  float theta = 0.0;
  while (current_l1 >  (1.0 + epsilon) * lambda ||
	 current_l1 < lambda) {
    theta = (max + min) / 2.0;
    current_l1 = 0.0;
    for (unsigned int i = 0; i < non_zeros.size(); ++i) {
      current_l1 += fmax(0, non_zeros[i] - theta);
    }
    if (current_l1 <= lambda) {
      max = theta;
    } else {
      min = theta;
    }
  }

  for (int i = 0; i < dimensions_; ++i) {
    if (weights_[i] > 0) weights_[i] = fmax(0, weights_[i] - theta);
    if (weights_[i] < 0) weights_[i] = fmin(0, weights_[i] + theta);
  } 
}


void SfWeightVector::ProjectToL1Ball(float lambda) {
  // Bail out early if possible.
  float current_l1 = 0.0;
  for (int i = 0; i < dimensions_; ++i) {
    if (fabsf(ValueOf(i)) > 0) current_l1 += fabsf(ValueOf(i));
  }
  if (current_l1 < lambda) return;

  vector<int> workspace_a;
  vector<int> workspace_b;
  vector<int> workspace_c;
  vector<int>* U = &workspace_a;
  vector<int>* L = &workspace_b;
  vector<int>* G = &workspace_c;
  vector<int>* temp;
  // Populate U with all non-zero elements in weight vector.
  for (int i = 0; i < dimensions_; ++i) {
    if (fabsf(ValueOf(i)) > 0) {
      U->push_back(i);
      current_l1 += fabsf(ValueOf(i));
    }
  }

  // Find the value of theta.
  double partial_pivot = 0;
  double partial_sum = 0;
  while (U->size() > 0) {
    G->clear();
    L->clear();
    int k = (*U)[static_cast<int>(rand() % U->size())];
    float pivot_k = fabsf(ValueOf(k));
    float partial_sum_delta = fabsf(ValueOf(k));
    float partial_pivot_delta = 1.0;
    // Partition U using pivot_k.
    for (unsigned int i = 0; i < U->size(); ++i) {
      float w_i = fabsf(ValueOf((*U)[i]));
      if (w_i >= pivot_k) {
	if ((*U)[i] != k) {
	  partial_sum_delta += w_i;
	  partial_pivot_delta += 1.0;
	  G->push_back((*U)[i]);
	}
      } else {
	L->push_back((*U)[i]);
      }
    }
    if ((partial_sum + partial_sum_delta) -
	pivot_k * (partial_pivot + partial_pivot_delta) < lambda) {
      partial_sum += partial_sum_delta;
      partial_pivot += partial_pivot_delta;
      temp = U;
      U = L;
      L = temp;
    } else {
      temp = U;
      U = G;
      G = temp;
    }
  }

  // Perform the projection.
  float theta = (partial_sum - lambda) / partial_pivot;  
  squared_norm_ = 0.0;
  for (int i = 0; i < dimensions_; ++i) {
    if (ValueOf(i) == 0.0) continue;
    int sign = (ValueOf(i) > 0) ? 1 : -1;
    weights_[i] = sign * fmax((sign * ValueOf(i) - theta), 0); 
    squared_norm_ += weights_[i] * weights_[i];
  }
  scale_ = 1.0;
}


//-----------------------------------------------------------------//
//---------------- SfWeightVector Private Methods ----------------//
//-----------------------------------------------------------------//

void SfWeightVector::ScaleToOne() {
  for (int i = 0; i < dimensions_; ++i) {
    weights_[i] *= scale_;
  }
  scale_ = 1.0;
}
