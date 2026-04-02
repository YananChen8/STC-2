import json, tqdm, pysubs2, os, argparse


def main():
    parser = argparse.ArgumentParser(description="Add subtitles to VideoMME JSONL.")
    parser.add_argument("--input", "-i", required=True, help="Path to input videomme.jsonl file.")
    parser.add_argument("--subtitle_dir", required=True, help="Directory containing *.srt subtitle files.")
    parser.add_argument("--output", "-o", required=True, help="Path to save output JSONL with subtitles.")
    args = parser.parse_args()

    lines = open(args.input).readlines()
    with open(args.output, 'w') as f:
        for line in tqdm.tqdm(lines):
            datum = json.loads(json.loads(line))
            srt_path = os.path.join(args.subtitle_dir, datum['videoID'] + '.srt')
            if os.path.exists(srt_path):
                subs = pysubs2.load(srt_path, encoding="utf-8")
                subtitles = []
                for sub in subs:
                    sub_text = sub.text.replace("\\N", " ")
                    if sub_text.strip():
                        subtitles.append(sub_text)
                subtitles = " ".join(subtitles)
            else:
                subtitles = ""
            datum['subtitles'] = subtitles
            f.write(json.dumps(datum) + '\n')


if __name__ == "__main__":
    main()
