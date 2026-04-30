#!/usr/bin/env python3
"""
baseline1_tfidf.py - Pure TF-IDF Baseline Matcher
------------------------------------------------
Saves results to JSON file (same format as cv_matcher.py)

Usage:
python baseline1_tfidf.py --cv ./cv-jsons/Fahim_Hoque.json  --index ./csv-encoder/job_index --output tfidf_matches.json
"""

import argparse
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Baseline1TFIDF:
    """TF-IDF Baseline Matcher"""
    
    def __init__(self):
        self.name = "TF-IDF Baseline"
    
    def match(self, cv_path, index_prefix, top_n=10, output_path=None):
        """
        Match CV to jobs and save results to JSON
        """
        # Load CV
        with open(cv_path, 'r', encoding='utf-8') as f:
            cv = json.load(f)
        
        # Extract CV text
        cv_text = self._extract_cv_text(cv)
        cv_name = cv.get('name', 'Unknown')
        
        # Load jobs
        meta_path = f"{index_prefix}_metadata.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            job_metadata = json.load(f)
        
        # Build job texts
        job_texts = [self._build_job_text(job) for job in job_metadata]
        
        # TF-IDF matching
        all_texts = [cv_text] + job_texts
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        cv_vector = tfidf_matrix[0:1]
        job_vectors = tfidf_matrix[1:]
        similarities = cosine_similarity(cv_vector, job_vectors)[0]
        
        # Build results (SAME FORMAT as cv_matcher.py)
        results = []
        for i, (job, sim) in enumerate(zip(job_metadata, similarities)):
            results.append({
                "job_id": job.get("job_id"),
                "job_title": job.get("job_title", "Unknown"),
                "company": job.get("company", "Unknown"),
                "location": job.get("location", "Unknown"),
                "salary_range": job.get("salary_range", "Not specified"),
                "deadline": job.get("deadline", "Not specified"),
                "final_score": round(sim * 100, 2),
                "breakdown": {
                    "method": "TF-IDF",
                    "similarity": round(sim, 4)
                }
            })
        
        # Sort and limit
        results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = results[:top_n]
        
        # Save to JSON (CRITICAL: This is how compare_all.py gets data)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(top_results, f, ensure_ascii=False, indent=2)
            print(f"✓ Results saved to: {output_path}")
        
        # Print preview
        self._print_results(cv_name, top_results)
        
        return top_results
    
    def _extract_cv_text(self, cv):
        """Extract ALL text from CV"""
        text_parts = []
        if cv.get('name'): text_parts.append(cv['name'])
        if cv.get('summary'): text_parts.append(cv['summary'])
        if cv.get('skills'): text_parts.append(' '.join(cv['skills']))
        
        for edu in cv.get('education', []):
            if edu.get('degree'): text_parts.append(edu['degree'])
        
        exp = cv.get('experience', [])
        entries = exp if isinstance(exp, list) else exp.get('entries', [])
        for e in entries:
            if e.get('title'): text_parts.append(e['title'])
            if e.get('description'): text_parts.append(e['description'][:300])
        
        return ' '.join(text_parts).lower()
    
    def _build_job_text(self, job):
        """Build job text"""
        parts = [
            job.get('job_title', ''),
            job.get('job_description', '')[:600],
            job.get('skills_required', ''),
            job.get('education_requirements', '')[:100],
        ]
        return ' '.join(p for p in parts if p).lower()
    
    def _print_results(self, cv_name, results):
        print("\n" + "="*65)
        print(f"TF-IDF RESULTS for {cv_name}")
        print("="*65)
        for i, r in enumerate(results[:10], 1):
            print(f"\n#{i} [{r['final_score']:.1f}%] {r['job_title']}")
            print(f"    Company: {r['company']}")


def main():
    parser = argparse.ArgumentParser(description="TF-IDF Baseline Matcher")
    parser.add_argument("--cv", required=True, help="CV JSON file path")
    parser.add_argument("--index", default="job_index", help="Job index prefix")
    parser.add_argument("--top", type=int, default=10, help="Top N matches")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()
    
    matcher = Baseline1TFIDF()
    matcher.match(args.cv, args.index, args.top, args.output)


if __name__ == "__main__":
    main()