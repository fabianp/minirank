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
// sf-sparse-vector.h
//
// Author: D. Sculley, December 2008
// dsculley@google.com or dsculley@cs.tufts.edu
//
// A sparse vector for use with sofia-ml.  Vector elements are contained
// in an stl vector, stored in a struct containing (feature id, feature value)
// pairs.  Feature id's are assumed to be unique, sorted, and strictly positive.
// Sparse vector is assumed to be in svm-light format, with the distinction
// that the class label may be a float rather than an integer, and
// that there is an optional group id value.
//
// <label> <group id?> <feature>:<value> ... <feature><value> <#comment?>
//
// The group id is set by a <prefix>:<id number>, such as qid:3.  For
// our purposes, the actual prefix string is ignored.  The only requirement
// is that it begins with a letter in [a-zA-Z].
//
// The comment string is optional.  If it is needed, use a # character
// to begin the comment.  The remainder of the string is then treated
// as the comment.
//
// Note that features must be sorted in ascending order, by feature id.
// Also, feature id 0 is reserved for the bias term.

#ifndef SF_SPARSE_VECTOR_H__
#define SF_SPARSE_VECTOR_H__

#include <float.h>
#include <string>
#include <vector>

#define SF_UNDEFINED_VAL FLT_MAX

using std::string;
using std::vector;

// Each element of the SfSparseVector is represented as a FeatureValuePair.
// Bundling these as a struct improves memory locality.
struct FeatureValuePair {
  int id_;
  float value_;
};

class SfSparseVector {
 public:
  // Construct a new vector from a string.  Input format is svm-light format:
  // <label> <feature>:<value> ... <feature:value> # comment<\n>
  // No bias term is used.
  SfSparseVector(const char* in_string);

  // Constructs a new vector from a string, as above, but also sets the bias
  // term to 1 iff use_bias_term is set to true.
  SfSparseVector(const char* in_string, bool use_bias_term);

  // Construct a new vector that is the difference of two vectors, (a - b).
  // This is useful for ranking problems, etc.
  SfSparseVector(const SfSparseVector& a, const SfSparseVector& b, float y);

  // Returns a string-format representation of the vector, in svm-light format.
  string AsString() const;

  // Methods for interacting with features
  inline int NumFeatures() const { return features_.size(); }
  inline int FeatureAt(int i) const { return features_[i].id_; }
  inline float ValueAt(int i) const { return features_[i].value_; }

  // Getters and setters.
  void SetY(float new_y) { y_ = new_y; }
  void SetA(float new_a) { a_ = new_a; }
  void SetGroupId(const string& new_id) { group_id_ = new_id; }
  void SetComment(const string& new_comment) { comment_ = new_comment; }
  float GetY() const { return y_; }
  float GetA() const { return a_; }
  float GetSquaredNorm() const { return squared_norm_; }
  const string& GetGroupId() const { return group_id_; }
  const string& GetComment() const { return comment_; }

  // Adds a new (id, value) FeatureValuePair to the end of the vector, and
  // updates the internal squared_norm_ member.
  void PushPair (int id, float value);

  // Clear all feature values and the cached squared_norm_, leaving all
  // other information unchanged.
  void ClearFeatures() { features_.clear(); squared_norm_ = 0; }

 private:
  void AddToSquaredNorm(float addend) { squared_norm_ += addend; }

  // Common initialization method shared by constructors, adding vector data
  // by parsing a string in SVM-light format.
  void Init(const char* in_string);

  // Sets up the bias term, indexed by feature id 0.
  void SetBias() { PushPair(0, 1); }

  // Sets up the bias term as null value, indexed by feature id 0.
  void NoBias() { PushPair(0, 0); }

  // Exits if the input format of the file is incorrect.
  void DieFormat(const string& reason);

  // Members.
  // Typically, only non-zero valued features are stored.  This vector is assumed
  // to hold feature id, feature value pairs in order sorted by feature id.  The
  // special feature id 0 is always set to 1, encoding bias.
  vector<FeatureValuePair> features_;

  // y_ is the class label.  We store this as a float, rather than an int,
  // so that this class may be used for regression problems, etc., if desired.
  float y_;

  // a_ is the current alpha value in optimization.
  float a_;

  // squared_norm_ = x1*x1 + ... + xN*xN
  float squared_norm_;

  // Use this member when examples belong to distinct groups.  For instance,
  // in ranking problems examples are grouped by query id.  By default,
  // this is set to 0.
  string group_id_;

  // comment_ can be any string-based comment.
  string comment_;
};

#endif // SF_SPARSE_VECTOR_H__
