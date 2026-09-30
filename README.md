# Image Matching POC

One-week POC investigating:

> Can a free/open-source multi-signal image matching pipeline
> substantially increase recall compared with a single matching technique?

## Dataset generation

Place raw images in:

    raw_files/

Then run:

    python3 src/dataset_builder.py raw_files/

The script creates:

    data/originals/
    data/variants/
    data/metadata/
    data/ground_truth/
    evaluation/

No image matching is performed by the dataset builder.

Ground truth generated from controlled transformations is automatic.

Human judgement is represented using TODO fields.
