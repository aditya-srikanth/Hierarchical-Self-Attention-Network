"""
Module to implement hyper parameter tuning
"""
import sys
import json
import itertools
from copy import deepcopy
from data_generator import generate_data
from average_calculator import average_calculator
from params import params as lstm_params
from cnn_params import params as cnn_params

lstm_params = deepcopy(lstm_params)
cnn_params = deepcopy(cnn_params)
lstm_params_opt = {
  "lr": None,
  "lr_decay": None
}


cnn_params_opt = {
  "lstm_size": None
}
best_fscore = 0
best_configuration = None
best_results = None


def merge_dicts(dict1, dict2):
  """
  utility function to merge two dictionaries and return a new dictionary,
  without modifying any dict.
  Args:
    dict1 (dict): dictionary to be merged
    dict2 (dict): dictionary to be merged

  Returns:
    dict : merged dictionary
  """
  z = dict1.copy()
  z.update(dict2)
  return z

if __name__ == "__main__":
  if sys.argv[1] == "lstm":
    params = lstm_params
    params_opt = lstm_params_opt
  elif sys.argv[1] == "cnn":
    params = cnn_params
    params_opt = cnn_params_opt
  params_values = {
    "lstm_size" : [25, 50, 75, 100]
  }

  for key in params_values:
    del(params[key])

  combinations = list(itertools.product(
    params_values["lstm_size"]
  ))
  i = 1
  j = len(combinations)
  with open("cnn_hybrid_fasttext100d_rest_lstm_size_tuning_results.txt", "w") as fp:
    for lstm_size in combinations:
      print("{} out of {} combinations done".format(i-1, j))
      i += 1
      params_opt["lstm_size"] = lstm_size
      use_params = merge_dicts(params, params_opt)
      print("*******************************")
      print("Configuation:")
      print(json.dumps(params_opt, indent=2))
      if i == 2:
        generate_data(use_params)
      results = average_calculator(model=sys.argv[1], k=10, gen_data=False,
                                   kwargs=use_params)
      f_score = results["fscore"]
      fp.write("********************\n")
      if f_score > best_fscore:
        best_fscore = f_score
        best_configuration = use_params
        best_results = results
        fp.write("Best configuration so far!\n")
        print("Best configuration so far!\n")
      fp.write(json.dumps(results, indent=2))
      fp.write("\nConfiguation:\n")
      fp.write(json.dumps(use_params, indent=2))
      fp.write("\n*********************\n")
      print("Results:")
      print(json.dumps(results, indent=2))
      print("*********************************")

    fp.write("Best configuration: \n")
    fp.write(json.dumps(best_configuration, indent=2))
    fp.write("\nwith score:\n")
    fp.write(json.dumps(best_results, indent=2))
    fp.close()
  print("Tuning done!")
  print("Best configuration: ")
  print(json.dumps(best_configuration, indent=2))
  print("With Score:")
  print(json.dumps(best_results, indent=2))
  
