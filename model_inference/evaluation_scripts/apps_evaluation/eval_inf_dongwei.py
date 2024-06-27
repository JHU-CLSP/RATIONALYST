"""
Run solutions from one problem.
"""
import argparse
import json
import numpy as np
import os
import pprint
import multiprocessing
import testing_util_inf as test_util

# for timing debugging
from datetime import datetime, date
from tqdm import tqdm

from types import SimpleNamespace
from typing import Dict



EXAMPLE_RESULTS = {"0": [[-2]],"1": [[False,False,False]],"2": [[True,True]],"3": [[False,True,False,True,False,False,False,True,False,True,False,True,True,True,False,True]],"4": [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]}
EXAMPLE_ARGS = SimpleNamespace(debug=True)
TIMEOUT = 50

def print_results(results: dict, args: argparse.Namespace=None):
    """
    Given the results evaluated against the testcases we output some statistics.
    Results are saved to a text file.
    """
    file_path = args.results_loc
    res = []
    per_prob_res = []
    all_correct = []
    all = []
    for index in range(len(results)):
        problem_results = np.asarray(results[index])
        res.extend([problem_results])
        per_prob_res.append(np.mean(problem_results > 0))
        all_correct.append(np.all(problem_results > 0))

    total_testcases = len(res)
    compile_errors = len([e for e in res if -2 in e])
    runtime_errors = len([e for e in res if -1 in e])

    with open(file_path, 'w') as text_file:
        # if args and args.debug:
        #     text_file.write(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases }\n")
        #     text_file.write(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}\n")
        #     text_file.write(f"number of test cases run(Question) = {total_testcases}\n")
        text_file.write(f"number of compile errors = {compile_errors} avg = {compile_errors / total_testcases }\n")
        text_file.write(f"number of runtime errors = {runtime_errors} avg = {runtime_errors / total_testcases}\n")
        text_file.write(f"number of test cases run(Question) = {total_testcases}\n")

        text_file.write(f"Test Case Average (average accuracy over problems) = {np.mean(per_prob_res)}\n")
        text_file.write(f"Strict Accuracy (all test cases passed / total problems) = {np.mean(all_correct)}\n")

    print(f"Results saved to {file_path}")

def check_correctness(code, input_output, timeout, debug):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(prob_path, generation, debug, result):
        result.append(test_util.run_test(code, input_output, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(code, input_output, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        # Ideally we would consider that all tests failed but we can't access the number of tests here easily
        # So we use 21=the average number of tests for a sample in the test split instead
        avg_number_tests = 21
        result = [[-1] * avg_number_tests]
        if debug:
            print(f"global timeout")
    return result[0]

def eval_and_save_problems(args):
    # with open(args.test_loc, "r") as f:
    #     problems = sorted(json.load(f))
    problems = [i for i in range(args.start, args.end)] 
    
    f = [json.loads(line) for line in open(args.root)]
    codes = {index: f[int(index)]['full_response'] for index in problems}
    input_outputs = {index: f[int(index)]['input_output'] for index in problems}
    
    res = []
    results = {}
    for id in problems:
        if args.debug:
            print(f"\nTesting solution {4000 + id}")
        curr_res = [-2]
        try:
            curr_res = check_correctness(code=codes[id], input_output=input_outputs[id], timeout=TIMEOUT, debug=args.debug)
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                print(f"Results were not all True: {curr_res}")
        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_res, list)
            res.append(curr_res)
    if args.debug:
            print(f"\nHow to read results [-2] = compile error, [-1] = runtime error [False] = failed test case [True] = passed test case")
    
    
    results = res
    
    results_loc = args.save + "/all_results_dongwei_700.json"
    with open(results_loc, "w") as f:
        try:
            f.write(json.dumps(results))
        except Exception as e:
            import pdb; pdb.set_trace()
            print("didn't save problem due to {e}")

    return results

def main(args):
    start_time = datetime.now()  # Start the timer

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    results = {}
    if args.print_results:
        results = {}
        codes_loc = os.path.join(args.save, f"all_codes.json")
        if os.path.exists(codes_loc):
            results_loc = os.path.join(args.save, f"all_results.json") 
        else:
            results_loc = os.path.join(args.save, f"{args.start}-{args.end}_results.json") 
        with open(results_loc, "r") as f: 
            results = json.load(f)
    else:
        results = eval_and_save_problems(args)

    print_results(results, args)

    end_time = datetime.now()  # End the timer
    total_time = end_time - start_time
    print(f"Total running time: {total_time}")

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    parser = argparse.ArgumentParser(description="Testing a Language Model on Python Code")
    # parser.add_argument("-t","--test_loc", default="/weka/scratch/djiang21/nlp_proj/path_collection.json", type=str, help="path to the json containing problem paths to be evaluated.")
    
    parser.add_argument("-r","--root", default=f"/weka/scratch/djiang21/nlp_proj/model/model_inference/output_inf_700.json", type=str, help="where the data is stored.")
    # parser.add_argument("-r","--root", default=f"/weka/scratch/djiang21/nlp_proj/model/model_inference/output_700_world_model.json", type=str, help="where the data is stored.")
    # parser.add_argument("-r","--root", default=f"/weka/scratch/djiang21/nlp_proj/model/model_inference/output_dongwei_700.json", type=str, help="where the data is stored.")
    
    parser.add_argument("-s","--start", default=0, type=int)
    parser.add_argument("-e","--end", default=700, type=int, help="If you want to evaluate a subset of problems specify start and ending index. File with start and ending prefix must exist typically used with batch evaluation.")
    
    # parser.add_argument("-i", "--index", default=0, type=int)
    parser.add_argument("-p", "--print_results", action="store_true", help="If you have already evaluated the results and only want to print them.")
    parser.add_argument("-d", "--debug", default=False, action="store_true")
    parser.add_argument("--save", type=str, default="/weka/scratch/djiang21/nlp_proj/model/model_inference/", help="Where the evaluated data is loaded from and results saved to.")
    parser.add_argument("--stop-early", default=None, type=int)
    parser.add_argument("--results_loc", default='/weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/model_inference/apps_evaluation/output.txt', type=str)


    args = parser.parse_args()
    main(args)
