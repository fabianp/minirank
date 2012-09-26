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
#include <assert.h>
#include <iostream>

#include "sofia-ml-methods.h"

int main (int argc, char** argv) {
  SfSparseVector x_p("1.0 0:1 1:1");
  SfSparseVector x_n("-1.0 0:-1 1:-1");

  SfWeightVector perceptron_2(2);
  assert(sofia_ml::SingleMarginPerceptronStep(x_p, 1.0, 1, &perceptron_2));
  assert(!sofia_ml::SingleMarginPerceptronStep(x_p, 1.0, 1, &perceptron_2));
  assert(sofia_ml::SingleSvmPrediction(x_p, perceptron_2) == 2.0);
  assert(sofia_ml::SingleSvmPrediction(x_n, perceptron_2) == -2.0);

  SfWeightVector pegasos_2(2);
  assert(sofia_ml::SinglePegasosStep(x_p, 1.0, 0.1, &pegasos_2));
  assert(pegasos_2.ValueOf(0) == 1.0);
  assert(pegasos_2.ValueOf(0) == 1.0);
  
  SfWeightVector pegasos_3(2);
  SfSparseVector x_p2("1.0 0:100");
  assert(sofia_ml::SinglePegasosStep(x_p2, 1.0, 0.1, &pegasos_3));
  assert(pegasos_3.ValueOf(0) < 3.16 + 0.01);
  assert(pegasos_3.ValueOf(0) > 3.16 - 0.01);

  SfWeightVector rank_2(2);
  assert(sofia_ml::SinglePegasosRankStep(x_p2, x_n, 1.0, 0.1, &rank_2));
  assert(!sofia_ml::SinglePegasosRankStep(x_n, x_n, 1.0, 0.1, &rank_2));
  assert(rank_2.ValueOf(0) < 2.84 + 0.01);
  assert(rank_2.ValueOf(0) > 2.84 - 0.01);

  SfDataSet data_set_2(false);
  data_set_2.AddVector("1 1:1.0 2:1.0");
  data_set_2.AddVector("-1 1:-1.0 2:-1.0");
  data_set_2.AddVector("1 1:0.5 2:-1.0");
  data_set_2.AddVector("-1 1:-1.0 2:-1.0");
  SfWeightVector pegasos_5(3);
  
  srand(100);
  sofia_ml::StochasticOuterLoop(data_set_2,
				sofia_ml::PEGASOS,
				sofia_ml::PEGASOS_ETA,
				0.1,
				0,
				4,
				&pegasos_5);
  assert(pegasos_5.ValueOf(1) < 0.560);
  assert(pegasos_5.ValueOf(1) > 0.559);
  assert(pegasos_5.ValueOf(2) < 0.560);
  assert(pegasos_5.ValueOf(2) > 0.559);

  vector<float> predictions;
  sofia_ml::SvmPredictionsOnTestSet(data_set_2, pegasos_5, &predictions);
  assert(predictions.size() == 4);
  assert(predictions[0] < 1.119);
  assert(predictions[0] > 1.118);
  assert(predictions[2] < -0.27);
  assert(predictions[2] > -0.28);

  float svm_objective = sofia_ml::SvmObjective(data_set_2, pegasos_5, 0.1);

  float expected_objective = 
    ((0.559 * 0.559 + 0.559 * 0.559) * 0.1 / 2.0) +  // weight vector penalty
    (0.0 + 0.0 + 1.27 + 0.0) / 4;  // loss penalty

  assert (svm_objective < expected_objective + 0.01);
  assert (svm_objective > expected_objective - 0.01);

  std::cout << argv[0] << ": PASS" << std::endl;
}
