import torch
import sys, os
from multiprocessing import Process
from gumbel_softmax import *
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

torch.set_float32_matmul_precision('highest')

NUM_PROCS = 16
NUM_MSGS = 2500


def test_runner_hcrl(proc_id, model, model_path, output_dir, output_name):
    ids = [258+i for i in range(proc_id, 27, NUM_PROCS)]
    for i in ids:
        for j in range(2, 10, 2):
            results = run_gsm_multitoken_HCRL(model, model_path, i, [i for i in range(9-j)], j, NUM_MSGS)
            with open(output_dir + "/" + output_name + str(proc_id) + ".txt", 'a') as f:
                f.write("ID " + str(i) + ", " + str(j) + " tokens:\n")
                f.write(str(results))
                f.write("\n")
                
def test_runner_cic(proc_id, model, model_path, output_dir, output_name):
    ids = [257+i for i in range(proc_id, 72, NUM_PROCS)]
    for i in ids:
        for j in range(2, 10, 2):
            results = run_gsm_multitoken_CIC(model, model_path, i, [i for i in range(9-j)], j, NUM_MSGS)
            with open(output_dir + "/" + output_name + str(proc_id) + ".txt", 'a') as f:
                f.write("ID " + str(i) + ", " + str(j) + " tokens:\n")
                f.write(str(results))
                f.write("\n")
    
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Saved off model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to test, hcrl or cic")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to outputs")
    parser.add_argument("--exp_name", type=str, required=True, help="Outputs name")
    parser.add_argument("--num_procs", type=int, default=NUM_PROCS, required=False, help="Number of processes to run the attack, each process will have its own GPU stream")
    parser.add_argument("--num_msgs", type=int, default=2500, required=False, help="Number of messages to consider")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if args.dataset.lower() == 'hcrl':
        targ = test_runner_hcrl
    elif args.dataset.lower() == 'cic':
        targ = test_runner_cic
    else:
        print("unknown dataset!")
        sys.exit()
        
    futures = []
    with ProcessPoolExecutor(max_workers=NUM_PROCS) as pool:
        for i in range(NUM_PROCS):
            futures.append(pool.submit(targ, i,args.model, args.model_path, args.output_dir, args.exp_name))
        wait(futures, return_when=ALL_COMPLETED)
    
if __name__ == "__main__":
    main()