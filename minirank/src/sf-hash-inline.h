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
// sf-hash-inline.h
//
// This is a port of Bob Jenkins's one-at-a-time-hash, modified
// to hash int's rather than char's.  In particular, we're interested
// having fast methods for exactly one key, and for two or more keys.
// Because of this small number of keys, we perform the first two steps
// of the hash twice on the first key considered.
//
// Note that a key of 0 returns a hash of 0.

#ifndef SF_HASH_INLINE_H__
#define SF_HASH_INLINE_H__

#include <vector>
using std::vector;

// Hash Function for a single int.
unsigned int SfHash(int key, int mask);

// Hash function for two int's.
unsigned int SfHash(int key_1, int key_2, int mask);

// Hash function for a vector of int's.  Assumes that keys has
// at least one entry.
unsigned int SfHash(const vector<int>& keys, int mask);

// Construct a mask with 1's in the num_bits lowest order bits.
int SfHashMask(int num_bits);

#endif  // SF_HASH_INLINE_H__
