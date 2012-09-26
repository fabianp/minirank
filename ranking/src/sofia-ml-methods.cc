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
// sofia-ml-methods.cc
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Implementation of sofia-ml-methods.h

#include "sofia-ml-methods.h"

#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

// The MIN_SCALING_FACTOR is used to protect against combinations of
// lambda * eta > 1.0, which will cause numerical problems for regularization
// and PEGASOS projection.  
#define MIN_SCALING_FACTOR 0.0000001

namespace sofia_ml {
  
  // --------------------------------------------------- //
  //         Helper functions (Not exposed in API)
  // --------------------------------------------------- //

  int RandInt(int num_vals) {
    return static_cast<int>(rand()) % num_vals;
  }

  float RandFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
  }

  const SfSparseVector& RandomExample(const SfDataSet& data_set) {
    int num_examples = data_set.NumExamples();
    int i = static_cast<int>(rand()) % num_examples;
    if (i < 0) {
      i += num_examples;
    }
    return data_set.VectorAt(i);
  }

  inline float GetEta (EtaType eta_type, float lambda, int i) {
    switch (eta_type) {
    case BASIC_ETA:
      return 10.0 / (i + 10.0);
      break;
    case PEGASOS_ETA:
      return 1.0 / (lambda * i);
      break;
    case CONSTANT:
      return 0.02;
      break;
    default:
      std::cerr << "EtaType " << eta_type << " not supported." << std::endl;
      exit(0);
    }
    std::cerr << "Error in GetEta, we should never get here." << std::endl;
    return 0;
  }
  
  // --------------------------------------------------- //
  //            Stochastic Loop Strategy Functions
  // --------------------------------------------------- //

  void StochasticOuterLoop(const SfDataSet& training_set,
			   LearnerType learner_type,
			   EtaType eta_type,
			   float lambda,
			   float c,
			   int num_iters,
			   SfWeightVector* w) {
    for (int i = 1; i <= num_iters; ++i) {
      int random_example = static_cast<int>(rand()) % training_set.NumExamples();
      const SfSparseVector& x = training_set.VectorAt(random_example);
      float eta = GetEta(eta_type, lambda, i);
      OneLearnerStep(learner_type, x, eta, c, lambda, w);
    }
  }  

  void BalancedStochasticOuterLoop(const SfDataSet& training_set,
				   LearnerType learner_type,
				   EtaType eta_type,
				   float lambda,
				   float c,
				   int num_iters,
				   SfWeightVector* w) {
    // Create index of positives and negatives for fast sampling
    // of disagreeing pairs.
    vector<int> positives;
    vector<int> negatives;
    for (int i = 0; i < training_set.NumExamples(); ++i) {
      if (training_set.VectorAt(i).GetY() > 0.0)
	positives.push_back(i);
      else
	negatives.push_back(i);
    }

    // For each iteration, randomly sample one positive and one negative and
    // take one gradient step for each.
    for (int i = 1; i <= num_iters; ++i) {
      float eta = GetEta(eta_type, lambda, i);

      const SfSparseVector& pos_x =
	training_set.VectorAt(positives[RandInt(positives.size())]);
      OneLearnerStep(learner_type, pos_x, eta, c, lambda, w);

      const SfSparseVector& neg_x =
	training_set.VectorAt(negatives[RandInt(negatives.size())]);
      OneLearnerStep(learner_type, neg_x, eta, c, lambda, w);
    }
  }

  void StochasticRocLoop(const SfDataSet& training_set,
			 LearnerType learner_type,
			 EtaType eta_type,
			 float lambda,
			 float c,
			 int num_iters,
			 SfWeightVector* w) {
    // Create index of positives and negatives for fast sampling
    // of disagreeing pairs.
    vector<int> positives;
    vector<int> negatives;
    for (int i = 0; i < training_set.NumExamples(); ++i) {
      if (training_set.VectorAt(i).GetY() > 0.0)
	positives.push_back(i);
      else
	negatives.push_back(i);
    }

    // For each step, randomly sample one positive and one negative and
    // take a pairwise gradient step.
    for (int i = 1; i <= num_iters; ++i) {
      float eta = GetEta(eta_type, lambda, i);
      const SfSparseVector& pos_x =
	training_set.VectorAt(positives[RandInt(positives.size())]);
      const SfSparseVector& neg_x =
	training_set.VectorAt(negatives[RandInt(negatives.size())]);
      OneLearnerRankStep(learner_type, pos_x, neg_x, eta, c, lambda, w);
    }
  }

  void StochasticClassificationAndRocLoop(const SfDataSet& training_set,
					   LearnerType learner_type,
					   EtaType eta_type,
					   float lambda,
					   float c,
					   float rank_step_probability,
					   int num_iters,
					   SfWeightVector* w) {
    // Create index of positives and negatives for fast sampling
    // of disagreeing pairs.
    vector<int> positives;
    vector<int> negatives;
    for (int i = 0; i < training_set.NumExamples(); ++i) {
      if (training_set.VectorAt(i).GetY() > 0.0)
	positives.push_back(i);
      else
	negatives.push_back(i);
    }

    for (int i = 1; i <= num_iters; ++i) {
      float eta = GetEta(eta_type, lambda, i);
      if (RandFloat() < rank_step_probability) {
	// For each step, randomly sample one positive and one negative and
	// take a pairwise gradient step.
	const SfSparseVector& pos_x =
	  training_set.VectorAt(positives[RandInt(positives.size())]);
	const SfSparseVector& neg_x =
	  training_set.VectorAt(negatives[RandInt(negatives.size())]);
	OneLearnerRankStep(learner_type, pos_x, neg_x, eta, c, lambda, w);
      } else {
	// Take a classification step.
	int random_example =
	  static_cast<int>(rand()) % training_set.NumExamples();
	const SfSparseVector& x = training_set.VectorAt(random_example);
	float eta = GetEta(eta_type, lambda, i);
	OneLearnerStep(learner_type, x, eta, c, lambda, w);      
      }
    }
  }
  
  void StochasticClassificationAndRankLoop(const SfDataSet& training_set,
					   LearnerType learner_type,
					   EtaType eta_type,
					   float lambda,
					   float c,
					   float rank_step_probability,
					   int num_iters,
					   SfWeightVector* w) {
    std::map<string, std::map<float, vector<int> > > group_id_y_to_index;
    std::map<string, int> group_id_y_to_count;
    for (int i = 0; i < training_set.NumExamples(); ++i) {
      const string& group_id = training_set.VectorAt(i).GetGroupId();
      group_id_y_to_index[group_id][training_set.VectorAt(i).GetY()].push_back(i);
      group_id_y_to_count[group_id] += 1;
    }
    
    for (int i = 1; i <= num_iters; ++i) {
      if (RandFloat() < rank_step_probability) {
	// Take a rank step.
	const SfSparseVector& a = RandomExample(training_set);
	const string& group_id = a.GetGroupId();
	float a_y = a.GetY();
	const std::map<float, vector<int> >& y_to_list =
	  group_id_y_to_index[group_id];
	int range = group_id_y_to_count[group_id] -
	  group_id_y_to_index[group_id][a_y].size();
	if (range == 0) continue;
	unsigned int random_int = RandInt(range);
	for (std::map<float, vector<int> >::const_iterator iter =
	       y_to_list.begin();
	     iter != y_to_list.end();
	     iter++) {
	  if (iter->first == a_y) continue;
	  if (random_int < iter->second.size()) {
	    const SfSparseVector& b = 
	      training_set.VectorAt((iter->second)[random_int]);
	    float eta = GetEta(eta_type, lambda, i);
	    OneLearnerRankStep(learner_type, a, b, eta, c, lambda, w);
	    break;
	  }
	  random_int -= iter->second.size();
	}
      } else {
	// Take a classification step.
	int random_example =
	  static_cast<int>(rand()) % training_set.NumExamples();
	const SfSparseVector& x = training_set.VectorAt(random_example);
	float eta = GetEta(eta_type, lambda, i);
	OneLearnerStep(learner_type, x, eta, c, lambda, w);
      }
    }
  }

  void StochasticRankLoop(const SfDataSet& training_set,
			 LearnerType learner_type,
			 EtaType eta_type,
			 float lambda,
			 float c,
			 int num_iters,
			 SfWeightVector* w) {
    std::map<string, std::map<float, vector<int> > > group_id_y_to_index;
    std::map<string, int> group_id_y_to_count;
    for (int i = 0; i < training_set.NumExamples(); ++i) {
      const string& group_id = training_set.VectorAt(i).GetGroupId();
      group_id_y_to_index[group_id][training_set.VectorAt(i).GetY()].push_back(i);
      group_id_y_to_count[group_id] += 1;
    }
    
    for (int i = 1; i <= num_iters; ++i) {
      const SfSparseVector& a = RandomExample(training_set);
      const string& group_id = a.GetGroupId();
      float a_y = a.GetY();
      const std::map<float, vector<int> >& y_to_list =
	group_id_y_to_index[group_id];
      int range =
	group_id_y_to_count[group_id] - group_id_y_to_index[group_id][a_y].size();
      if (range == 0) continue;
      unsigned int random_int = RandInt(range);
      for (std::map<float, vector<int> >::const_iterator iter = y_to_list.begin();
	   iter != y_to_list.end();
	   iter++) {
	if (iter->first == a_y) continue;
	if (random_int < iter->second.size()) {
	  const SfSparseVector& b = 
	    training_set.VectorAt((iter->second)[random_int]);
	  float eta = GetEta(eta_type, lambda, i);
	  OneLearnerRankStep(learner_type, a, b, eta, c, lambda, w);
	  break;
	}
	random_int -= iter->second.size();
      }
    }
  }

  void StochasticQueryNormRankLoop(const SfDataSet& training_set,
				   LearnerType learner_type,
				   EtaType eta_type,
				   float lambda,
				   float c,
				   int num_iters,
				   SfWeightVector* w) {
    // Create a map of group id's to examples.
    std::map<string, vector<int> > group_id_to_examples;
    for (int i = 0; i < training_set.NumExamples(); ++i) {
      const string& group_id = training_set.VectorAt(i).GetGroupId();
      group_id_to_examples[group_id].push_back(i);
    }

    // Assign each group id a unique integer id.
    std::map<int, vector<int>* > group_id_index;
    int i = 0;
    for (std::map<string, vector<int> >::iterator group_id_iter =
	   group_id_to_examples.begin();
	 group_id_iter != group_id_to_examples.end();
	 ++group_id_iter) {
      group_id_index[i] = &(group_id_iter->second);
      ++i;
    }

    for (int i = 1; i <= num_iters; ++i) {
      int group_id = RandInt(group_id_index.size());
      vector<int>* group_index = group_id_index[group_id];
      int group_index_size = group_index->size();
      const SfSparseVector& a =
	training_set.VectorAt((*group_index)[RandInt(group_index_size)]);

      bool found_differing_pair = false;
      int num_chances = 1000;
      while (!found_differing_pair && --num_chances > 0) {
	const SfSparseVector& b =
	  training_set.VectorAt((*group_index)[RandInt(group_index_size)]);

	found_differing_pair = a.GetY() != b.GetY();
        if (found_differing_pair) {
	  float eta = GetEta(eta_type, lambda, i);
	  OneLearnerRankStep(learner_type, a, b, eta, c, lambda, w);
	  break;
	}
      }
    } 
  }

  //------------------------------------------------------------------------------//
  //                    Methods for Applying a Model on Data                      //
  //------------------------------------------------------------------------------//

  float SingleSvmPrediction(const SfSparseVector& x,
			    const SfWeightVector& w) {
    return w.InnerProduct(x);
  }

  float SingleLogisticPrediction(const SfSparseVector& x,
				 const SfWeightVector& w) {
    float p = w.InnerProduct(x);
    return exp(p) / (1.0 + exp(p));
  }
  
  void SvmPredictionsOnTestSet(const SfDataSet& test_data,
			       const SfWeightVector& w,
			       vector<float>* predictions) {
    predictions->clear();
    int size = test_data.NumExamples();
    for (int i = 0; i < size; ++i) {
      predictions->push_back(w.InnerProduct(test_data.VectorAt(i)));
    }
  }

  void LogisticPredictionsOnTestSet(const SfDataSet& test_data,
				    const SfWeightVector& w,
				    vector<float>* predictions) {
    predictions->clear();
    int size = test_data.NumExamples();
    for (int i = 0; i < size; ++i) {
      predictions->push_back(SingleLogisticPrediction(test_data.VectorAt(i),
						      w));
    }
  }

  float SvmObjective(const SfDataSet& data_set,
		     const SfWeightVector& w,
		     float lambda) {
    vector<float> predictions;
    SvmPredictionsOnTestSet(data_set, w, &predictions);
    float objective = w.GetSquaredNorm() * lambda / 2.0;
    for (int i = 0; i < data_set.NumExamples(); ++i) {
      float loss_i = 1.0 - (predictions[i] * data_set.VectorAt(i).GetY());
      float incremental_loss = (loss_i < 0.0) ? 
	0.0 : loss_i / data_set.NumExamples();
      objective += incremental_loss;
    }
    return objective;
  }

  // --------------------------------------------------- //
  //       Single Stochastic Step Strategy Methods
  // --------------------------------------------------- //
  
  bool OneLearnerStep(LearnerType learner_type,
		      const SfSparseVector& x,
		      float eta,
		      float c,
		      float lambda,
		      SfWeightVector* w) {
    switch (learner_type) {
    case PEGASOS:
      return SinglePegasosStep(x, eta, lambda, w);
    case MARGIN_PERCEPTRON:
      return SingleMarginPerceptronStep(x, eta, c, w);
    case PASSIVE_AGGRESSIVE:
      return SinglePassiveAggressiveStep(x, lambda, c, w);
    case LOGREG_PEGASOS:
      return SinglePegasosLogRegStep(x, eta, lambda, w);
    case LOGREG:
      return SingleLogRegStep(x, eta, lambda, w);
    case LMS_REGRESSION:
      return SingleLeastMeanSquaresStep(x, eta, lambda, w);
    case SGD_SVM:
      return SingleSgdSvmStep(x, eta, lambda, w);
    case ROMMA:
      return SingleRommaStep(x, w);
    default:
      std::cerr << "Error: learner_type " << learner_type
		<< " not supported." << std::endl;
      exit(0);
    }
  }

  bool OneLearnerRankStep(LearnerType learner_type,
			  const SfSparseVector& a,
			  const SfSparseVector& b,
			  float eta,
			  float c,
			  float lambda,
			  SfWeightVector* w) {
    switch (learner_type) {
    case PEGASOS:
      return SinglePegasosRankStep(a, b, eta, lambda, w);
    case MARGIN_PERCEPTRON:
      return SingleMarginPerceptronRankStep(a, b, eta, c, w);
    case PASSIVE_AGGRESSIVE:
      return SinglePassiveAggressiveRankStep(a, b, lambda, c, w);
    case LOGREG_PEGASOS:
      return SinglePegasosLogRegRankStep(a, b, eta, lambda, w);
    case LOGREG:
      return SingleLogRegRankStep(a, b, eta, lambda, w);
    case LMS_REGRESSION:
      return SingleLeastMeanSquaresRankStep(a, b, eta, lambda, w);
    case SGD_SVM:
      return SingleSgdSvmRankStep(a, b, eta, lambda, w);
    case ROMMA:
      return SingleRommaRankStep(a, b, w);
    default:
      std::cerr << "Error: learner_type " << learner_type
		<< " not supported." << std::endl;
      exit(0);
    }
  }

  // --------------------------------------------------- //
  //            Single Stochastic Step Functions
  // --------------------------------------------------- //
  
  bool SinglePegasosStep(const SfSparseVector& x,
			  float eta,
			  float lambda,
			  SfWeightVector* w) {
    float p = x.GetY() * w->InnerProduct(x);    

    L2Regularize(eta, lambda, w);    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.GetY() != 0.0) {
      w->AddVector(x, (eta * x.GetY())); 
    }

    PegasosProjection(lambda, w);
    return (p < 1.0 && x.GetY() != 0.0);
  }

  bool SingleRommaStep(const SfSparseVector& x,
		       SfWeightVector* w) {
    float wx = w->InnerProduct(x);
    float p = x.GetY() * wx;
    const float kVerySmallNumber = 0.0000000001;
    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.GetY() != 0.0) {
      float xx = x.GetSquaredNorm();
      float ww = w->GetSquaredNorm();
      float c = ((xx * ww) - p + kVerySmallNumber) /
	((xx * ww) - (wx * wx) + kVerySmallNumber);

      float d = (ww * (x.GetY() - wx) + kVerySmallNumber) /
	((xx * ww) - (wx * wx) + kVerySmallNumber);

      // Avoid numerical problems caused by examples of extremely low magnitude.
      if (c >= 0.0) {
	w->ScaleBy(c);
	w->AddVector(x, d); 
      }
    }

    return (p < 1.0 && x.GetY() != 0.0);
  }

  bool SingleSgdSvmStep(const SfSparseVector& x,
			  float eta,
			  float lambda,
			  SfWeightVector* w) {
    float p = x.GetY() * w->InnerProduct(x);    

    L2Regularize(eta, lambda, w);    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p < 1.0 && x.GetY() != 0.0) {
      w->AddVector(x, (eta * x.GetY())); 
    }

    return (p < 1.0 && x.GetY() != 0.0);
  }

  bool SingleMarginPerceptronStep(const SfSparseVector& x,
				  float eta,
				  float c,
				  SfWeightVector* w) {
    if (x.GetY() * w->InnerProduct(x) <= c) {
      w->AddVector(x, eta * x.GetY());
      return true;
    } else {
      return false;
    }
  }

  bool SinglePegasosLogRegStep(const SfSparseVector& x,
			       float eta,
			       float lambda,
			       SfWeightVector* w) {
    float loss = x.GetY() / (1 + exp(x.GetY() * w->InnerProduct(x)));

    L2Regularize(eta, lambda, w);    
    w->AddVector(x, (eta * loss));
    PegasosProjection(lambda, w);
    return (true);
  }

  bool SingleLogRegStep(const SfSparseVector& x,
			float eta,
			float lambda,
			SfWeightVector* w) {
    float loss = x.GetY() / (1 + exp(x.GetY() * w->InnerProduct(x)));

    L2Regularize(eta, lambda, w);    
    w->AddVector(x, (eta * loss));
    return (true);
  }

  bool SingleLeastMeanSquaresStep(const SfSparseVector& x,
				  float eta,
				  float lambda,
				  SfWeightVector* w) {
    float loss = x.GetY() - w->InnerProduct(x);
    L2Regularize(eta, lambda, w);    
    w->AddVector(x, (eta * loss));
    PegasosProjection(lambda, w);
    return (true);
  }

  bool SinglePassiveAggressiveStep(const SfSparseVector& x,
				   float lambda,
				   float max_step,
				   SfWeightVector* w) {
    float p = 1 - (x.GetY() * w->InnerProduct(x));    
    // If x has non-zero loss, perform gradient step in direction of x.
    if (p > 0.0 && x.GetY() != 0.0) {
      float step = p / x.GetSquaredNorm();
      if (step > max_step) step = max_step;
      w->AddVector(x, (step * x.GetY())); 
    }

    if (lambda > 0.0) {
      PegasosProjection(lambda, w);
    }
    return (p < 1.0 && x.GetY() != 0.0);
  }

  bool SinglePassiveAggressiveRankStep(const SfSparseVector& a,
				       const SfSparseVector& b,
				       float lambda,
				       float max_step,
				       SfWeightVector* w) {
    float y = (a.GetY() > b.GetY()) ? 1.0 :
      (a.GetY() < b.GetY()) ? -1.0 : 0.0;
    float p = 1 - (y * w->InnerProductOnDifference(a, b)); 
    // If (a-b) has non-zero loss, perform gradient step in direction of x.
    if (p > 0.0 && y != 0.0) {
      // Compute squared norm of (a-b).
      int i = 0;
      int j = 0;
      float squared_norm = 0;
      while (i < a.NumFeatures() || j < b.NumFeatures()) {
	int a_feature = (i < a.NumFeatures()) ? a.FeatureAt(i) : INT_MAX;
	int b_feature = (j < b.NumFeatures()) ? b.FeatureAt(j) : INT_MAX;
	if (a_feature < b_feature) {
	  squared_norm += a.ValueAt(i) * a.ValueAt(i);
	  ++i;
	} else if (b_feature < a_feature) {
	  squared_norm += b.ValueAt(j) * b.ValueAt(j);
	  ++j;
	} else {
	  squared_norm += (a.ValueAt(i) - b.ValueAt(j)) * (a.ValueAt(i) - b.ValueAt(j));
	  ++i;
	  ++j;
	}
      }
      float step = p / squared_norm;
      if (step > max_step) step = max_step;
      w->AddVector(a, (step * y)); 
      w->AddVector(b, (step * y * -1.0)); 
    }

    if (lambda > 0.0) {
      PegasosProjection(lambda, w);
    }
    return (p > 0 && y != 0.0);
  }

  bool SinglePegasosRankStep(const SfSparseVector& a,
			     const SfSparseVector& b,
			     float eta,
			     float lambda,
			     SfWeightVector* w) {
    float y = (a.GetY() > b.GetY()) ? 1.0 :
      (a.GetY() < b.GetY()) ? -1.0 : 0.0;
    float p = y * w->InnerProductOnDifference(a, b);

    L2Regularize(eta, lambda, w);

    // If (a - b) has non-zero loss, perform gradient step.         
    if (p < 1.0 && y != 0.0) {
      w->AddVector(a, (eta * y));
      w->AddVector(b, (-1.0 * eta * y));
    }

    PegasosProjection(lambda, w);
    return (p < 1.0 && y != 0.0);
  }

  bool SingleSgdSvmRankStep(const SfSparseVector& a,
			     const SfSparseVector& b,
			     float eta,
			     float lambda,
			     SfWeightVector* w) {
    float y = (a.GetY() > b.GetY()) ? 1.0 :
      (a.GetY() < b.GetY()) ? -1.0 : 0.0;
    float p = y * w->InnerProductOnDifference(a, b);

    L2Regularize(eta, lambda, w);

    // If (a - b) has non-zero loss, perform gradient step.         
    if (p < 1.0 && y != 0.0) {
      w->AddVector(a, (eta * y));
      w->AddVector(b, (-1.0 * eta * y));
    }

    return (p < 1.0 && y != 0.0);
  }

  bool SingleLeastMeanSquaresRankStep(const SfSparseVector& a,
			     const SfSparseVector& b,
			     float eta,
			     float lambda,
			     SfWeightVector* w) {
    float y = (a.GetY() - b.GetY());
    float loss = y - w->InnerProductOnDifference(a, b);

    L2Regularize(eta, lambda, w);
    w->AddVector(a, (eta * loss));
    w->AddVector(b, (-1.0 * eta * loss));
    PegasosProjection(lambda, w);
    return (true);
  }

  bool SingleRommaRankStep(const SfSparseVector& a,
			   const SfSparseVector& b,
			   SfWeightVector* w) {
    // Not the most efficient approach, but it takes care of
    // computing the squared norm of x with minimal coding effort.
    float y = (a.GetY() > b.GetY()) ? 1.0 :
      (a.GetY() < b.GetY()) ? -1.0 : 0.0;
    SfSparseVector x_diff(a, b, y);
    if (y != 0.0) {
      return SingleRommaStep(x_diff, w);
    } else {
      return false;
    }
  }

  bool SinglePegasosLogRegRankStep(const SfSparseVector& a,
				   const SfSparseVector& b,
				   float eta,
				   float lambda,
				   SfWeightVector* w) {
    float y = (a.GetY() > b.GetY()) ? 1.0 :
      (a.GetY() < b.GetY()) ? -1.0 : 0.0;
    float loss = y / (1 + exp(y * w->InnerProductOnDifference(a, b)));
    L2Regularize(eta, lambda, w);    

    w->AddVector(a, (eta * loss));
    w->AddVector(b, (-1.0 * eta * loss));

    PegasosProjection(lambda, w);
    return (true);
  }

  bool SingleLogRegRankStep(const SfSparseVector& a,
			    const SfSparseVector& b,
			    float eta,
			    float lambda,
			    SfWeightVector* w) {
    float y = (a.GetY() > b.GetY()) ? 1.0 :
      (a.GetY() < b.GetY()) ? -1.0 : 0.0;
    float loss = y / (1 + exp(y * w->InnerProductOnDifference(a, b)));
    L2Regularize(eta, lambda, w);    

    w->AddVector(a, (eta * loss));
    w->AddVector(b, (-1.0 * eta * loss));
    return (true);
  }

  bool SingleMarginPerceptronRankStep(const SfSparseVector& a,
				      const SfSparseVector& b,
				      float eta,
				      float c,
				      SfWeightVector* w) {
    float y = (a.GetY() > b.GetY()) ? 1.0 :
      (a.GetY() < b.GetY()) ? -1.0 : 0.0;
    if (y * w->InnerProductOnDifference(a, b) <= c) {
      w->AddVector(a, eta);
      w->AddVector(b, -1.0 * eta);
      return true;
    } else {
      return false;
    }
  }

  bool SinglePegasosRankWithTiesStep(const SfSparseVector& rank_a,
				     const SfSparseVector& rank_b,
				     const SfSparseVector& tied_a,
				     const SfSparseVector& tied_b,
				     float eta,
				     float lambda,
				     SfWeightVector* w) {
    float rank_y = (rank_a.GetY() > rank_b.GetY()) ? 1.0 :
      (rank_a.GetY() < rank_b.GetY()) ? -1.0 : 0.0;
    float rank_p = rank_y * w->InnerProductOnDifference(rank_a, rank_b);
    float tied_p = w->InnerProductOnDifference(tied_a, tied_b);

    L2Regularize(eta, lambda, w);

    // If (rank_a - rank_b) has non-zero loss, perform gradient step.         
    if (rank_p < 1.0 && rank_y != 0.0) {
      w->AddVector(rank_a, (eta * rank_y));
      w->AddVector(rank_b, (-1.0 * eta * rank_y));
    }

    // The value of tied_p should ideally be 0.0.  We penalize with squared
    // loss for predictions away from 0.0.
    if (tied_a.GetY() == tied_b.GetY()) {
      w->AddVector(tied_a, (eta * (0.0 - tied_p)));
      w->AddVector(tied_b, (-1.0 * eta * (0.0 - tied_p)));
    }

    PegasosProjection(lambda, w);
    return (true);
  }

  void L2Regularize(float eta, float lambda, SfWeightVector* w) {
    float scaling_factor = 1.0 - (eta * lambda);
    if (scaling_factor > MIN_SCALING_FACTOR) {
      w->ScaleBy(1.0 - (eta * lambda));  
    } else {
      w->ScaleBy(MIN_SCALING_FACTOR); 
    }
  }

  void L2RegularizeSeveralSteps(float eta,
				float lambda,
				float effective_steps,
				SfWeightVector* w) {
    float scaling_factor = 1.0 - (eta * lambda);
    scaling_factor = pow(scaling_factor, effective_steps);
    if (scaling_factor > MIN_SCALING_FACTOR) {
      w->ScaleBy(1.0 - (eta * lambda));  
    } else {
      w->ScaleBy(MIN_SCALING_FACTOR); 
    }
  }
  
  void PegasosProjection(float lambda, SfWeightVector* w) {
    float projection_val = 1 / sqrt(lambda * w->GetSquaredNorm());
    if (projection_val < 1.0) {
      w->ScaleBy(projection_val);
    }
  }
  
}  // namespace sofia_ml
  
