#!/usr/bin/env python3
"""
baseline2_jaccard.py - Pure Jaccard Similarity Baseline
------------------------------------------------------
COMPLETELY STANDALONE - Only skill overlap, no semantics
Saves results to JSON file (same format as cv_matcher.py)

Usage:
    python baseline2_jaccard.py --cv cv.json --index job_index --output jaccard_results.json --top 10
"""

import argparse
import json
import re


class Baseline2Jaccard:
    """Jaccard Baseline - Pure skill overlap only"""
    
    def __init__(self):
        self.name = "Jaccard Baseline"
        self.description = "Pure skill overlap with no weighting"
        
        # Core skills for matching (comprehensive list from your cv_matcher.py)
        self.core_skills = {
            "python", "java", "javascript", "typescript", "c++", "c#", "sql",
            "dart", "flutter", "react", "node.js", "spring boot", "django",
            "fastapi", "flask", "machine learning", "deep learning", "nlp",
            "computer vision", "ai", "docker", "kubernetes", "aws", "azure",
            "gcp", "mongodb", "postgresql", "git", "rest api", "tensorflow",
            "pytorch", "scikit-learn", "go", "kotlin", "swift", "php", "ruby",
            "scala", "redis", "firebase", "android", "ios", "jetpack compose",
            "selenium", "jenkins", "linux", "bash", "r", "matlab", "vue",
            "angular", "express", "next.js"
        }
    
    def _extract_cv_skills(self, cv):
        """Extract skills from CV"""
        skills = cv.get('skills', [])
        # Also extract skills from experience descriptions
        exp_skills = self._extract_skills_from_experience(cv)
        
        cv_set = set([s.lower().strip() for s in skills if s])
        cv_set.update(exp_skills)
        
        return cv_set
    
    def _extract_skills_from_experience(self, cv):
        """Extract skills from experience descriptions"""
        exp = cv.get('experience', [])
        entries = exp if isinstance(exp, list) else exp.get('entries', [])
        
        found_skills = set()
        for e in entries:
            desc = e.get('description', '').lower()
            for skill in self.core_skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', desc):
                    found_skills.add(skill)
        
        return found_skills
    
    def _extract_job_skills(self, job):
        """Extract skills from job description (simple substring match)"""
        job_text = (
            job.get('job_title', '') + ' ' +
            job.get('job_description', '') + ' ' +
            job.get('skills_required', '')
        ).lower()
        
        job_skills = set()
        for skill in self.core_skills:
            # Simple substring match with word boundaries
            if re.search(r'\b' + re.escape(skill) + r'\b', job_text):
                job_skills.add(skill)
        
        return job_skills
    
    def match(self, cv_path, index_prefix, top_n=10, output_path=None):
        """
        Match CV to jobs using Jaccard similarity on skills
        
        Args:
            cv_path: Path to CV JSON file
            index_prefix: Prefix of job index (e.g., 'job_index')
            top_n: Number of top matches to return
            output_path: Path to save results JSON
        
        Returns:
            List of top matching jobs with scores
        """
        print("\n" + "="*65)
        print("BASELINE 2: JACCARD SIMILARITY MATCHER")
        print("="*65)
        
        # Load CV
        print("\n[1/4] Loading CV...")
        with open(cv_path, 'r', encoding='utf-8') as f:
            cv = json.load(f)
        
        cv_skills = self._extract_cv_skills(cv)
        cv_name = cv.get('name', 'Unknown')
        print(f"    Candidate: {cv_name}")
        print(f"    Skills found: {len(cv_skills)}")
        
        # Load jobs
        print("\n[2/4] Loading job index...")
        meta_path = f"{index_prefix}_metadata.json"
        with open(meta_path, 'r', encoding='utf-8') as f:
            job_metadata = json.load(f)
        print(f"    Jobs loaded: {len(job_metadata)}")
        
        # Match each job
        print("\n[3/4] Computing Jaccard similarities...")
        results = []
        
        for i, job in enumerate(job_metadata):
            job_skills = self._extract_job_skills(job)
            
            if job_skills:
                # Jaccard similarity = intersection / union
                intersection = len(cv_skills & job_skills)
                union = len(cv_skills | job_skills)
                score = (intersection / union) * 100 if union > 0 else 0
                
                # Partial match bonus (for substring matches)
                partial = 0
                for cs in cv_skills:
                    for js in job_skills:
                        if cs != js and (cs in js or js in cs):
                            partial += 1
                partial_bonus = min(partial / max(len(job_skills), 1), 0.2)
                score = min(score + (partial_bonus * 100), 100)
            else:
                score = 0
            
            results.append({
                "job_id": job.get("job_id"),
                "job_title": job.get("job_title", "Unknown"),
                "company": job.get("company", "Unknown"),
                "location": job.get("location", "Unknown"),
                "salary_range": job.get("salary_range", "Not specified"),
                "deadline": job.get("deadline", "Not specified"),
                "final_score": round(score, 2),
                "breakdown": {
                    "method": "Jaccard",
                    "cv_skills_count": len(cv_skills),
                    "job_skills_count": len(job_skills),
                    "matched_skills": len(cv_skills & job_skills),
                    "jaccard_score": round(score, 2)
                }
            })
            
            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"    Processed {i + 1}/{len(job_metadata)} jobs...")
        
        # Sort and limit
        print("\n[4/4] Sorting and saving results...")
        results.sort(key=lambda x: x["final_score"], reverse=True)
        top_results = results[:top_n]
        
        # Save to JSON (same format as cv_matcher.py)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(top_results, f, ensure_ascii=False, indent=2)
            print(f"✓ Results saved to: {output_path}")
        
        # Print preview
        self._print_results(cv_name, top_results)
        
        # Print statistics
        print(f"\n{'='*65}")
        print("BASELINE SUMMARY")
        print("="*65)
        print(f"Method: Jaccard Similarity (Pure Skill Overlap)")
        print(f"Top score: {top_results[0]['final_score']:.1f}%")
        avg_score = sum(r['final_score'] for r in top_results) / len(top_results)
        print(f"Average top-{top_n} score: {avg_score:.1f}%")
        
        return top_results
    
    def _print_results(self, cv_name, results):
        """Print formatted results"""
        print("\n" + "="*65)
        print(f"TOP {len(results)} MATCHES FOR {cv_name.upper()} (Jaccard Baseline)")
        print("="*65)
        for i, r in enumerate(results[:10], 1):
            b = r.get('breakdown', {})
            print(f"\n#{i} [{r['final_score']:.1f}%] {r['job_title']}")
            print(f"    Company : {r['company']}")
            print(f"    Location: {r['location']}")
            print(f"    Skills matched: {b.get('matched_skills', 0)}/{b.get('job_skills_count', 0)}")


def main():
    parser = argparse.ArgumentParser(description="Jaccard Baseline Matcher")
    parser.add_argument("--cv", required=True, help="Path to CV JSON file")
    parser.add_argument("--index", default="job_index", help="Job index prefix")
    parser.add_argument("--top", type=int, default=10, help="Number of top matches")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()
    
    matcher = Baseline2Jaccard()
    matcher.match(args.cv, args.index, args.top, args.output)


if __name__ == "__main__":
    main()