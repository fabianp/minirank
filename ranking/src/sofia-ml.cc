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
// sofia-ml.cc
//
// Author: D. Sculley
// dsculley@google.com or dsculley@cs.tufts.edu
//
// Main file for stochastic active set svm (sofia-ml), 
// a variant of the PEGASOS stochastic gradient svm solver.

#include <assert.h>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "sf-hash-weight-vector.h"
#include "sofia-ml-methods.h"
#include "sf-weight-vector.h"
#include "simple-cmd-line-helper.h"

using std::string;

void CommandLine(int argc, char** argv) {
  AddFlag("--training_file", "File to be used for training.", string(""));
  AddFlag("--test_file", "File to be used for testing.", string(""));
  AddFlag("--results_file", "File to which to write predictions.", string(""));
  AddFlag("--model_in", "Read in a model from this file.", string(""));
  AddFlag("--model_out", "Write the model to this file.", string(""));
  AddFlag("--random_seed",
	  "When set to non-zero value, use this seed instead of seed from system clock.\n"
	  "    This can be useful for parameter tuning in cross-validation, as setting \n"
	  "    a seed by hand forces examples to be sampled in the same order.  However\n"
	  "    for actual training/test, this should never be used.\n"
	  "    Default: 0",
	  int(0));
  AddFlag("--lambda", 
	  "Value of lambda for svm regularization.\n"
	  "    Default: 0.1",
	  float(0.1));
  AddFlag("--iterations",
	  "Number of stochastic gradient steps to take.\n"
	  "    Default: 100000",
	  int(100000));
  AddFlag("--learner_type",
	  "Type of learner to use.\n"
	  "    Options are: pegasos, passive-aggressive, margin-perceptron, "
	  "romma, sgd-svm, least-mean-squares, logreg, and logreg-pegasos.\n"
	  "    Default: pegasos",
	  string("pegasos"));
  AddFlag("--eta_type",
	  "Type of update for learning rate to use.\n"
	  "    Options are: basic, pegasos, constant (0.02).\n"
	  "    Default: pegasos",
	  string("pegasos"));
  AddFlag("--loop_type",
	  "Type of loop to use for training, controlling how examples are selected.\n"
	  "    Options are: stochastic, balanced-stochastic, "
	  "roc, rank, query-norm-rank, combined-ranking, "
	  "combined-roc\n"
	  "    Default: stochastic",
	  string("stochastic"));
  AddFlag("--prediction_type",
	  "Type of predictions to compute on test data.\n"
	  "    Options are: linear, logistic\n"
	  "    Use linear for SVM predictions.\n"
	  "    Default: linear",
	  string("linear"));
  AddFlag("--rank_step_probability",
	  "Probability that we will take a rank step (as opposed to a  \n"
	  "    classification step) in a combined-ranking or "
	  "    combined-roc loop.\n"
	  "    Default: 0.5",
	  float(0.5));
  AddFlag("--passive_aggressive_c",
	  "Maximum size of any step taken in a single passive-aggressive update.",
	  float(10000000.0));
  AddFlag("--passive_aggressive_lambda",
	  "Lambda for pegasos-style projection for passive-aggressive update.\n"
	  "    When set to 0 (default) no projection is performed.",
	  float(0));
  AddFlag("--perceptron_margin_size",
	  "Width of margin for perceptron with margins.\n"
	  "    Default of 1 is equivalent to un-regularized SVM-loss.",
	  float(1.0));
  AddFlag("--training_objective",
	  "Compute value of objective function on training data, after training.\n"
	  "    Default is not to do this.",
	  bool(false));
  AddFlag("--buffer_mb",
	  "Size of buffer to use in reading/writing to files, in MB.\n"
	  "    Default: 40",
	  int(40));
  AddFlag("--dimensionality",
	  "Index value of largest feature index in training data set. \n"
	  "    Default: 2^17 = 131072",
	  int(2<<16));
  AddFlag("--hash_mask_bits",
	  "When set to a non-zero value, causes the use of a hashed weight vector\n"
	  "    with hashed cross product features.  The size of the hash table is set\n"
	  "    to 2^--hash_mask_bits.  The same value of this flag must be used for\n"
	  "    testing and training.\n"
	  "    Default value of 0 shows that hash cross products are not used.",
	  int(0));
  AddFlag("--no_bias_term",
	  "When set, causes a bias term x_0 to be set to 0 for every \n"
	  "    feature vector loaded from files, rather than the default of x_0 = 1.\n"
	  "    This is equivalent to forcing a decision threshold of exactly 0 to be used.\n"
	  "    Same setting of this flag should be used for training and testing. Note that\n"
	  "    this flag as no effect for rank and roc optimzation.\n"
	  "    Default: not set.",
	  bool(false));
  ParseFlags(argc, argv);
}

void PrintElapsedTime(clock_t start, const string& message) {
  float num_secs = static_cast<double>(clock() - start) / CLOCKS_PER_SEC;
  std::cout << message << num_secs << std::endl;
}

void SaveModelToFile(const string& file_name, SfWeightVector* w) {
  std::fstream model_stream;
  model_stream.open(file_name.c_str(), std::fstream::out);
  if (!model_stream) {
    std::cerr << "Error opening model output file " << file_name << std::endl;
    exit(1);
  }
  std::cerr << "Writing model to: " << file_name << std::endl;
  model_stream << w->AsString() << std::endl;
  model_stream.close();
  std::cerr << "   Done." << std::endl;
}

void LoadModelFromFile(const string& file_name, SfWeightVector** w) {
  if (*w != NULL) {
    delete *w;
  }

  std::fstream model_stream;
  model_stream.open(file_name.c_str(), std::fstream::in);
  if (!model_stream) {
    std::cerr << "Error opening model input file " << file_name << std::endl;
    exit(1);
  }

  std::cerr << "Reading model from: " << file_name << std::endl;
  string model_string;
  std::getline(model_stream, model_string);
  model_stream.close();
  std::cerr << "   Done." << std::endl;

  *w = new SfWeightVector(model_string);
  assert(*w != NULL);
}

void TrainModel (const SfDataSet& training_data, SfWeightVector* w) {
  clock_t train_start = clock();
  assert(w != NULL);

  // Default values.
  float lambda = CMD_LINE_FLOATS["--lambda"];
  float c = 0.0;

  sofia_ml::EtaType eta_type;
  if (CMD_LINE_STRINGS["--eta_type"] == "basic")
    eta_type = sofia_ml::BASIC_ETA;
  else if (CMD_LINE_STRINGS["--eta_type"] == "pegasos")
    eta_type = sofia_ml::PEGASOS_ETA;
  else if (CMD_LINE_STRINGS["--eta_type"] == "constant")
    eta_type = sofia_ml::CONSTANT;
  else {
    std::cerr << "--eta type " << CMD_LINE_STRINGS["--eta_type"] << " not supported.";
    exit(0);
  }
 
  sofia_ml::LearnerType learner_type;
  if (CMD_LINE_STRINGS["--learner_type"] == "pegasos")
    learner_type = sofia_ml::PEGASOS;
  else if (CMD_LINE_STRINGS["--learner_type"] == "margin-perceptron") {
    learner_type = sofia_ml::MARGIN_PERCEPTRON;
    c = CMD_LINE_FLOATS["--perceptron_margin_size"];
  }
  else if (CMD_LINE_STRINGS["--learner_type"] == "passive-aggressive") {
    learner_type = sofia_ml::PASSIVE_AGGRESSIVE;
    c = CMD_LINE_FLOATS["--passive_aggressive_c"];
    lambda = CMD_LINE_FLOATS["--passive_aggressive_lambda"];
  }
  else if (CMD_LINE_STRINGS["--learner_type"] == "logreg-pegasos")
    learner_type = sofia_ml::LOGREG_PEGASOS;
  else if (CMD_LINE_STRINGS["--learner_type"] == "logreg")
    learner_type = sofia_ml::LOGREG;
  else if (CMD_LINE_STRINGS["--learner_type"] == "least-mean-squares")
    learner_type = sofia_ml::LMS_REGRESSION;
  else if (CMD_LINE_STRINGS["--learner_type"] == "sgd-svm")
    learner_type = sofia_ml::SGD_SVM;
  else if (CMD_LINE_STRINGS["--learner_type"] == "romma")
    learner_type = sofia_ml::ROMMA;
  else {
    std::cerr << "--learner_type " << CMD_LINE_STRINGS["--learner_type"]
	      << " not supported.";
    exit(0);
  }
  
  if (CMD_LINE_STRINGS["--loop_type"] == "stochastic")
    sofia_ml::StochasticOuterLoop(training_data,
				learner_type,
				eta_type,
				lambda,
				c,
				CMD_LINE_INTS["--iterations"],
				w);
  else if (CMD_LINE_STRINGS["--loop_type"] == "balanced-stochastic")
    sofia_ml::BalancedStochasticOuterLoop(training_data,
					learner_type,
					eta_type,
					lambda,
					c,
					CMD_LINE_INTS["--iterations"],
					w);
  else if (CMD_LINE_STRINGS["--loop_type"] == "roc")
    sofia_ml::StochasticRocLoop(training_data,
			      learner_type,
			      eta_type,
			      lambda,
			      c,
			      CMD_LINE_INTS["--iterations"],
			      w);
  else if (CMD_LINE_STRINGS["--loop_type"] == "rank")
    sofia_ml::StochasticRankLoop(training_data,
			      learner_type,
			      eta_type,
			      lambda,
			      c,
			      CMD_LINE_INTS["--iterations"],
			      w);
  else if (CMD_LINE_STRINGS["--loop_type"] == "combined-ranking")
    sofia_ml::StochasticClassificationAndRankLoop(
		training_data,
		learner_type,
		eta_type,
		lambda,
		c,
		CMD_LINE_FLOATS["--rank_step_probability"],
		CMD_LINE_INTS["--iterations"],
		w);
  else if (CMD_LINE_STRINGS["--loop_type"] == "combined-roc")
    sofia_ml::StochasticClassificationAndRocLoop(
		training_data,
		learner_type,
		eta_type,
		lambda,
		c,
		CMD_LINE_FLOATS["--rank_step_probability"],
		CMD_LINE_INTS["--iterations"],
		w);
  else if (CMD_LINE_STRINGS["--loop_type"] == "query-norm-rank")
    sofia_ml::StochasticQueryNormRankLoop(training_data,
			      learner_type,
			      eta_type,
			      lambda,
			      c,
			      CMD_LINE_INTS["--iterations"],
			      w);
  else {
    std::cerr << "--loop_type " << CMD_LINE_STRINGS["--loop_type"] << " not supported.";
    exit(0);
  }

  PrintElapsedTime(train_start, "Time to complete training: ");
}

int main (int argc, char** argv) {
  CommandLine(argc, argv);
  
  if (CMD_LINE_INTS["--random_seed"] == 0) {
    srand(time(NULL));
  } else {
    std::cerr << "Using random_seed: " << CMD_LINE_INTS["--random_seed"] << std::endl;
    srand(CMD_LINE_INTS["--random_seed"]);
  }

  // Set up empty model with specified dimensionality.
  SfWeightVector* w  = NULL;
  if (CMD_LINE_INTS["--hash_mask_bits"] == 0) {
    w = new SfWeightVector(CMD_LINE_INTS["--dimensionality"]);
  } else {
    w = new SfHashWeightVector(CMD_LINE_INTS["--hash_mask_bits"]);
  }

  // Load model (overwriting empty model), if needed.
  if (!CMD_LINE_STRINGS["--model_in"].empty()) {
    LoadModelFromFile(CMD_LINE_STRINGS["--model_in"], &w); 
  }
  
  // Train model, if needed.
  if (!CMD_LINE_STRINGS["--training_file"].empty()) {
    std::cerr << "Reading training data from: " 
	      << CMD_LINE_STRINGS["--training_file"] << std::endl;
    clock_t read_data_start = clock();
    SfDataSet training_data(CMD_LINE_STRINGS["--training_file"],
			    CMD_LINE_INTS["--buffer_mb"],
			    !CMD_LINE_BOOLS["--no_bias_term"]);
    PrintElapsedTime(read_data_start, "Time to read training data: ");

    TrainModel(training_data, w);

    // Compute value of objective function on training data, if needed.
    if (CMD_LINE_BOOLS["--training_objective"]) {
      clock_t compute_objective_start = clock();
      float objective = sofia_ml::SvmObjective(training_data,
					      *w,
					      CMD_LINE_BOOLS["--lambda"]);
      PrintElapsedTime(compute_objective_start,
		       "Time to compute objective on training data: ");
      std::cout << "Value of objective function on training data after "
		<< CMD_LINE_INTS["--iterations"] << " iterations: "
		<< objective << std::endl;
    }
  }

  // Save model, if needed.
  if (!CMD_LINE_STRINGS["--model_out"].empty()) {
    SaveModelToFile(CMD_LINE_STRINGS["--model_out"], w);
  }
    
  // Test model on test data, if needed.
  if (!CMD_LINE_STRINGS["--test_file"].empty()) {
    std::cerr << "Reading test data from: " 
	      << CMD_LINE_STRINGS["--test_file"] << std::endl;
    clock_t read_data_start = clock();
    SfDataSet test_data(CMD_LINE_STRINGS["--test_file"],
			CMD_LINE_INTS["--buffer_mb"],
			!CMD_LINE_BOOLS["--no_bias_term"]);
    PrintElapsedTime(read_data_start, "Time to read test data: ");
    
    vector<float> predictions;
    clock_t predict_start = clock();
    if (CMD_LINE_STRINGS["--prediction_type"] == "linear")
      sofia_ml::SvmPredictionsOnTestSet(test_data, *w, &predictions);
    else if (CMD_LINE_STRINGS["--prediction_type"] == "logistic")
      sofia_ml::LogisticPredictionsOnTestSet(test_data, *w, &predictions);
    else {
      std::cerr << "--prediction " << CMD_LINE_STRINGS["--prediction_type"]
		<< " not supported.";
      exit(0);
    }

    PrintElapsedTime(predict_start, "Time to make test prediction results: ");
    
    std::fstream prediction_stream;
    prediction_stream.open(CMD_LINE_STRINGS["--results_file"].c_str(),
			   std::fstream::out);
    if (!prediction_stream) {
      std::cerr << "Error opening test results output file " 
		<< CMD_LINE_STRINGS["--results_file"] << std::endl;
      exit(1);
    }
    std::cerr << "Writing test results to: "
	      << CMD_LINE_STRINGS["--results_file"] << std::endl;
    for (unsigned int i = 0; i < predictions.size(); ++i) {
      prediction_stream << predictions[i] << "\t" 
			<< test_data.VectorAt(i).GetY() << std::endl;
    }
    prediction_stream.close();
    std::cerr << "   Done." << std::endl;
  }
}
