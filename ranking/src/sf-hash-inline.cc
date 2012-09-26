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
// sf-hash-inline.cc
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Implementation of ph-hash-inline.h

#include "sf-hash-inline.h"

// Hash Function for a single int.
unsigned int SfHash(int key, int mask) {
  int hash(key);
  hash += (hash << 10);
  hash ^= (hash >> 6);
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash & mask;
}

// Hash function for two int's.
unsigned int SfHash(int key_1, int key_2, int mask) {
  unsigned int hash(key_1);
  hash += (hash << 10);
  hash ^= (hash >> 6);

  hash += key_2;
  hash += (hash << 10);
  hash ^= (hash >> 6);

  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash & mask;
}

// Hash function for a vector of int's.  Assumes that keys has
// at least one entry.
unsigned int SfHash(const vector<int>& keys, int mask) {
  unsigned int hash = 0;
  for (unsigned int i = 0; i < keys.size(); ++i) {
    hash += keys[i];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }
  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);
  return hash & mask;
}

int SfHashMask(int num_bits) {
  int mask = 1;
  for (int i = 1; i < num_bits; ++i) {
    mask += (1 << i);
  }
  return mask;
}

