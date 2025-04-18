import torch
import sys, os
from multiprocessing import Process
from gumbel_softmax import *
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED

torch.set_float32_matmul_precision('highest')

NUM_PROCS = 16
NUM_MSGS = 2500
HCRL_FIRST_ID = 258 # the first id token value is 258
CIC_FIRST_ID = 257 # the first id token value is 257
            
def test_runner(f, id, model, model_path, output_dir, output_name):
    if f == run_gsm_multitoken_HCRL:
        id += HCRL_FIRST_ID
    elif f == run_gsm_multitoken_CIC:
        id += CIC_FIRST_ID
    for j in range(2, 10, 2):
        results = f(model, model_path, id, [i for i in range(9-j)], j, NUM_MSGS)
        with open(output_dir + "/" + output_name + str(id) + ".txt", 'a') as f:
            f.write("ID " + str(id) + ", " + str(j) + " tokens:\n")
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
    
    num_ids = 0
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    if args.dataset.lower() == 'hcrl':
        targ = run_gsm_multitoken_HCRL
        num_ids = 27
    elif args.dataset.lower() == 'cic':
        targ = run_gsm_multitoken_CIC
        num_ids = 72
    else:
        print("unknown dataset!")
        sys.exit()
        
    futures = []
    with ProcessPoolExecutor(max_workers=args.num_procs) as pool:
        for i in range(num_ids):
            futures.append(pool.submit(test_runner, targ, i,args.model, args.model_path, args.output_dir, args.exp_name))
        wait(futures, return_when=ALL_COMPLETED)
        for f in futures:
            print(f.result())
    
if __name__ == "__main__":
    main()