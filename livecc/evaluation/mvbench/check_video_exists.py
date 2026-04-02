import json, argparse
from utils.multiprocessor import local_mt


def check(line):
    datum = json.loads(line)
    if 'tvqa' in datum['video']:
        return line
    try:
        datum['video']
        return line
    except:
        print(datum['video'], 'not exists')


def main():
    parser = argparse.ArgumentParser(description="Filter MVBench JSONL to entries with existing videos.")
    parser.add_argument("--input", "-i", required=True, help="Path to input mvbench.jsonl file.")
    parser.add_argument("--output", "-o", required=True, help="Path to save filtered output JSONL.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers.")
    args = parser.parse_args()

    lines = open(args.input).readlines()
    existed_lines = local_mt(lines, check, desc='check', num_workers=args.num_workers)
    existed_lines = [line for line in existed_lines if line is not None]

    with open(args.output, 'w') as f:
        f.writelines(existed_lines)


if __name__ == "__main__":
    main()