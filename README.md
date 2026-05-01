# **RecruitBD-Comparisons**

#### This branch only holds manual comparisons files and folders. Compares the main used method for matching jobs with CV's with other baselines. 
---

## 🚀 **Commands**

#### Scrape jobs

```bash
cd jobs
python jobs.py

```
#### Build index
```bash
cd cv-matcher
cd csv-encoder
python build_index.py --jobs jobs.csv --output job_index

```
#### Run methods scripts serially with path
```bash
python cv_matcher.py --cv Raiyen_Zayed_Rakin_CV.json --index job_index --top 10 --output matches.json

# Run baselines
python baseline1_tfidf.py --cv Raiyen_Zayed_Rakin_CV.json --index job_index --output tfidf_results.json --top 10
python baseline2_jaccard.py --cv Raiyen_Zayed_Rakin_CV.json --index job_index --output jaccard_results.json --top 10

# Compare all
python compare_all.py --files matches.json tfidf_results.json jaccard_results.json