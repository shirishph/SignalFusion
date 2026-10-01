import sys
from pathlib import Path
import yaml

def main():
    if len(sys.argv) != 3:
        print("Usage: python src/demo.py <photo> <metadata.yaml>")
        sys.exit(1)

    photo_path = Path(sys.argv[1])
    metadata_path = Path(sys.argv[2])

    if not photo_path.exists():
        print(f"Photo not found: {photo_path}")
        sys.exit(1)

    if not metadata_path.exists():
        print(f"Metadata file not found: {metadata_path}")
        sys.exit(1)

    print(f"Photo:    {photo_path}")
    print(f"Metadata: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    timestamp = metadata["timestamp"]
    platform = metadata["platform"]

    print(f"Timestamp: {timestamp}")
    print(f"Platform:  {platform}")


if __name__ == "__main__":
    main()
