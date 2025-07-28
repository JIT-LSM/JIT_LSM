import argparse
import json
import os

from LargeModel.lib.MainLLM import MainLLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_file",type=str, required=True,
                        help="The input testing data file.")
    parser.add_argument("--project", type=str, required=True,
                        help="The input testing commit.")
    parser.add_argument("--commit",type=str, required=True,
                        help="The input testing commit.")
    parser.add_argument("--model_name",type=str,required=True,
                        help="Large language model name")
    parser.add_argument("--round", type=int,default=5,
                        help="The times to run")
    parser.add_argument("--seq",type=int,required=True,
                        help="the seq of running commit")
    parser.add_argument("--output_file", type=str, required=True,
                        help="The output testing data file.")
    parser.add_argument("--Cni",action='store_true',
                        help="Whether to run Check but no informed")
    parser.add_argument("--Ano", action='store_true',
                        help="Whether to run no assistance")
    parser.add_argument("--Cno", action='store_true',
                        help="Whether to run no checker")
    parser.add_argument("--Nfc", action='store_true',
                        help="Whether to run no function calling")
    parser.add_argument("--Full", action='store_true',
                        help="Whether to run JIT-LSM")

    args = parser.parse_args()
    return args

def main(args):
    data_path = f"{args.test_data_file}/{args.project}/{args.commit}/commit_data.json"
    with open(data_path,'r',encoding='utf-8') as f:
        commit_data = json.load(f)
    for r in range(args.round):
        check_path = f"{args.output_file}/{args.model_name}/full_{r+1}/{args.commit}.json"
        if args.Cni:
            print("[INFO]","当前模式是CNI")
            el = MainLLM(args.project,args.commit,commit_data,args.model_name,args.test_data_file,True,True,True,False)
        elif args.Ano:
            print("[INFO]", "当前模式是ANO")
            el = MainLLM(args.project, args.commit, commit_data, args.model_name,args.test_data_file, True, False, True, True)
        elif args.Cno:
            print("[INFO]", "当前模式是CNO")
            el = MainLLM(args.project, args.commit, commit_data, args.model_name,args.test_data_file, True, True, False, False)
        elif args.Nfc:
            print("[INFO]", "当前模式是NFC")
            el = MainLLM(args.project, args.commit, commit_data, args.model_name,args.test_data_file, False, False, False, False)
        elif args.Full:
            print("[INFO]", "当前模式是FULL")
            el = MainLLM(args.project, args.commit, commit_data, args.model_name,args.test_data_file, True, True, True, True)

        if os.path.exists(check_path):
            print("[LOG]",f"项目{args.project}第{args.seq}条数据的第{r + 1}轮已经跑过，将跳过")
            continue

        el.run()
        print("[LOG]", f"项目{args.project}第{args.seq}条数据的第{r + 1}轮跑完")
        #############
        el.write_message_to_json(r+1,args.seq,args.output_file)
        #############
        el._print_log()

if __name__ == '__main__':
    cur_args = parse_args()
    main(cur_args)