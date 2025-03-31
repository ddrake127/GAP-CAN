from gumbel_softmax import *
from multiprocessing import Process
import os

torch.set_float32_matmul_precision('highest')

NUM_PROCS = 16
NUM_MSGS = 2500

HCRL_SAVE_DIR = "test_results_hcrl"
CIC_SAVE_DIR = "test_results_cic"

def test_runner_hcrl(proc_id):
    ids = [258+i for i in range(proc_id, 27, NUM_PROCS)]
    for i in ids:
        for j in range(2, 10, 2):
            results = run_gsm_multitoken_HCRL(i, [i for i in range(9-j)], j, NUM_MSGS)
            with open(HCRL_SAVE_DIR + "/true_adv" + str(proc_id) + ".txt", 'a') as f:
                f.write("ID " + str(i) + ", " + str(j) + " tokens:\n")
                f.write(str(results))
                f.write("\n")
                
def test_runner_cic(proc_id):
    ids = [257+i for i in range(proc_id, 72, NUM_PROCS)]
    for i in ids:
        for j in range(2, 10, 2):
            results = run_gsm_multitoken_CIC(i, [i for i in range(9-j)], j, NUM_MSGS)
            with open(CIC_SAVE_DIR + "/true_adv" + str(proc_id) + ".txt", 'a') as f:
                f.write("ID " + str(i) + ", " + str(j) + " tokens:\n")
                f.write(str(results))
                f.write("\n")
    
    
def main():

    if not os.path.isdir(HCRL_SAVE_DIR):
        os.mkdir(HCRL_SAVE_DIR)
    if not os.path.isdir(CIC_SAVE_DIR):
        os.mkdir(CIC_SAVE_DIR)

    procs = []

    for i in range(NUM_PROCS):
        p = Process(target=test_runner_hcrl, args=(i,))
        procs.append(p)
        p.start()
    for i in range(NUM_PROCS):
        procs[i].join()
        
    procs = []
    for i in range(NUM_PROCS):
        p = Process(target=test_runner_cic, args=(i,))
        procs.append(p)
        p.start()
    for i in range(NUM_PROCS):
        procs[i].join()

    # log_coe, losses = run_gene([0,1,2,3,4,5,6,7], None)
    # mses = run_gsm([4], 263, 100, None)

    
    
    
    
if __name__ == "__main__":
    main()