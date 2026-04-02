import argparse
import json
from pathlib import Path

try:
    from demo.infer import LiveCCDemoInfer
except ModuleNotFoundError:
    from infer import LiveCCDemoInfer

def parse_args():
    parser = argparse.ArgumentParser(description="Run LiveCC CLI demo.")
    parser.add_argument("--model-path", default=None, help="HF model id, snapshot dir, or HF cache repo dir.")
    parser.add_argument("--video-path", default="demo/sources/howto_fix_laptop_mute_1080p.mp4")
    parser.add_argument("--message", default="Please describe the video.")
    parser.add_argument("--device", default=None, help="cuda, cpu, mps, or cuda:0 style device.")
    parser.add_argument("--max-seconds", type=int, default=31)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    video_path = args.video_path
    query = args.message
    output_path = args.output_path or f"demo/results/{Path(video_path).stem}.json"

    infer = LiveCCDemoInfer(model_path=args.model_path, device=args.device)
    state = {'video_path': video_path}
    commentaries = []
    t = 0
    for t in range(args.max_seconds):
        state['video_timestamp'] = t
        for (start_t, stop_t), response, state in infer.live_cc(
            message=query, state=state, 
            max_pixels = 384 * 28 * 28, repetition_penalty=1.05, 
            streaming_eos_base_threshold=0.0, streaming_eos_threshold_step=0
        ):
            print(f'{start_t}s-{stop_t}s: {response}')
            commentaries.append([start_t, stop_t, response])
        if state.get('video_end', False):
            break
        t += 1
    result = {'video_path': video_path, 'query': query, 'commentaries': commentaries}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"{video_path=}, {query=} => model_path={infer.model_path} => {output_path=}")
    json.dump(result, open(output_path, 'w'))
