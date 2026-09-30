# Evaluation Metrics

These metrics will be used in later experiments.

## Primary metric

### Recall@K

Of all known true matches, how many appear in the top K results?

Measure:

- Recall@1
- Recall@5
- Recall@10

## Secondary metrics

### Precision@K

Of the top K returned candidates, how many are true matches?

Measure:

- Precision@5
- Precision@10

### Mean Reciprocal Rank (MRR)

Measures how highly the first correct match appears.

### False Positive Rate

Measures incorrect matches against known negative relationships.

## Experimental comparison

Every matching experiment should record:

- Method
- Dataset
- Transformation
- Recall@1
- Recall@5
- Recall@10
- Precision@5
- Precision@10
- MRR
- False positives

The main hypothesis is concerned with whether combining multiple
independent signals improves recall compared with a single matching
technique.
