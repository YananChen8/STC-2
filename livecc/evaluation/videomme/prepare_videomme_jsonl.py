import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def read_subtitle(subtitle_path: Path) -> str | None:
    if not subtitle_path.exists():
        return None
    return subtitle_path.read_text(encoding="utf-8", errors="ignore").strip()


def main():
    parser = argparse.ArgumentParser(description="Convert local Video-MME parquet metadata to LiveCC jsonl.")
    parser.add_argument("--parquet_path", required=True, help="Path to Video-MME parquet file.")
    parser.add_argument("--video_dir", required=True, help="Directory containing *.mp4 videos.")
    parser.add_argument("--subtitle_dir", default=None, help="Directory containing *.srt subtitles.")
    parser.add_argument("--output_path", required=True, help="Output jsonl path.")
    parser.add_argument("--skip_missing_video", action="store_true", help="Skip samples whose local video file is missing.")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    subtitle_dir = Path(args.subtitle_dir) if args.subtitle_dir else None
    rows = pq.read_table(args.parquet_path).to_pylist()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as wf:
        for row in rows:
            video_path = video_dir / f"{row['videoID']}.mp4"
            if not video_path.exists():
                if args.skip_missing_video:
                    skipped += 1
                    continue
                raise FileNotFoundError(f"Missing local video file: {video_path}")

            datum = dict(row)
            datum["video"] = str(video_path)
            if subtitle_dir:
                subtitles = read_subtitle(subtitle_dir / f"{row['videoID']}.srt")
                if subtitles:
                    datum["subtitles"] = subtitles
            wf.write(json.dumps(datum, ensure_ascii=False) + "\n")
            written += 1

    print(f"written={written}, skipped={skipped}, output_path={output_path}")


if __name__ == "__main__":
    main()
