#!/usr/bin/env python3
"""
compare_all.py - Compare Results from Different Matchers
--------------------------------------------------------
Reads saved JSON files from each matcher and produces comprehensive comparison.

Usage:
    # First run all matchers to generate JSON files
    python cv_matcher.py --cv cv.json --index job_index --output my_results.json --top 10
    python baseline1_tfidf.py --cv cv.json --index job_index --output tfidf_results.json --top 10
    python baseline2_jaccard.py --cv cv.json --index job_index --output jaccard_results.json --top 10
    python baseline3_simple_weighted.py --cv cv.json --index job_index --output simple_results.json --top 10
    
    # Then compare
    python compare_all.py --files my_results.json tfidf_results.json jaccard_results.json
    python compare_all.py --files my_results.json tfidf_results.json jaccard_results.json --names "Our Method" "TF-IDF" "Jaccard" "Simple Weighted"
"""

import argparse
import json
import numpy as np
from tabulate import tabulate


def load_results(file_path):
    """Load results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_method_name_from_path(file_path, custom_names=None, index=0):
    """Extract method name from file path or use custom name"""
    if custom_names and index < len(custom_names):
        return custom_names[index]
    
    # Extract from filename
    import os
    base = os.path.basename(file_path)
    base = base.replace("_results.json", "").replace(".json", "")
    base = base.replace("_", " ").title()
    
    # Map common patterns
    name_map = {
        "my": "Our Full Method",
        "tfidf": "TF-IDF Baseline",
        "jaccard": "Jaccard Baseline",
        "simple": "Simple Weighted Baseline",
        "baseline1": "TF-IDF Baseline",
        "baseline2": "Jaccard Baseline",
        "baseline3": "Simple Weighted Baseline"
    }
    
    return name_map.get(base.lower(), base)


def compare_all_results(file_paths, method_names=None, top_n_display=10):
    """
    Compare results from multiple JSON files
    
    Args:
        file_paths: List of JSON file paths
        method_names: Optional list of method names
        top_n_display: Number of top matches to display in tables
    """
    
    print("\n" + "="*90)
    print("📊 COMPARATIVE EVALUATION: All Matching Methods")
    print("="*90)
    
    # Load all results
    all_results = {}
    method_names_list = []
    
    print("\n📁 Loading result files...")
    print("-" * 50)
    
    for i, file_path in enumerate(file_paths):
        try:
            results = load_results(file_path)
            name = get_method_name_from_path(file_path, method_names, i)
            all_results[name] = results
            method_names_list.append(name)
            print(f"✓ Loaded {len(results)} results from: {file_path} → {name}")
        except Exception as e:
            print(f"✗ Error loading {file_path}: {e}")
    
    if len(all_results) < 2:
        print("\n❌ Need at least 2 result files to compare. Exiting.")
        return
    
    # Determine which is "Your Method" (usually the one with highest top-1 score)
    your_method_name = None
    best_score = -1
    for name, results in all_results.items():
        if results and results[0]['final_score'] > best_score:
            best_score = results[0]['final_score']
            your_method_name = name
    
    print(f"\n🎯 Reference method (highest scoring): {your_method_name}")
    
    # ============================================================
    # TABLE 1: Top Matches Side by Side
    # ============================================================
    
    print("\n" + "="*90)
    print(f"📈 TABLE 1: Top {min(5, top_n_display)} Matches Per Method")
    print("="*90)
    
    # Prepare headers
    headers = ["Rank"] + list(all_results.keys())
    
    # Build table rows
    table_rows = []
    for rank in range(1, min(5, top_n_display) + 1):
        row = [f"#{rank}"]
        for method_name, results in all_results.items():
            if rank <= len(results):
                score = results[rank-1]['final_score']
                title = results[rank-1]['job_title']
                # Format with score and title
                row.append(f"{score:.1f}%\n{title[:30]}")
            else:
                row.append("-\n-")
        table_rows.append(row)
    
    print(tabulate(table_rows, headers=headers, tablefmt="grid", maxcolwidths=35))
    
    # ============================================================
    # TABLE 2: Score Statistics
    # ============================================================
    
    print("\n" + "="*90)
    print("📊 TABLE 2: Score Statistics")
    print("="*90)
    
    stats_headers = ["Method", "Top-1 Score", "Top-3 Avg", "Top-5 Avg", "Top-10 Avg", "Score Range"]
    stats_rows = []
    
    for method_name, results in all_results.items():
        if results:
            top1 = results[0]['final_score']
            top3_avg = np.mean([r['final_score'] for r in results[:3]])
            top5_avg = np.mean([r['final_score'] for r in results[:5]])
            top10_avg = np.mean([r['final_score'] for r in results[:10]])
            scores = [r['final_score'] for r in results]
            range_str = f"{min(scores):.1f} - {max(scores):.1f}"
            
            # Highlight best in each category
            top1_str = f"{top1:.1f}%"
            if top1 == max([all_results[m][0]['final_score'] for m in all_results if all_results[m]]):
                top1_str = f"🏆 {top1:.1f}%"
        else:
            top1 = top3_avg = top5_avg = top10_avg = 0
            range_str = "N/A"
            top1_str = "N/A"
        
        stats_rows.append([method_name, top1_str, f"{top3_avg:.1f}%", f"{top5_avg:.1f}%", f"{top10_avg:.1f}%", range_str])
    
    print(tabulate(stats_rows, headers=stats_headers, tablefmt="grid"))
    
    # ============================================================
    # TABLE 3: Improvement Analysis
    # ============================================================
    
    print("\n" + "="*90)
    print("🏆 TABLE 3: Improvement Analysis (vs Best Baseline)")
    print("="*90)
    
    # Find best baseline (excluding your method if identified)
    baseline_names = [n for n in all_results.keys() if n != your_method_name]
    
    best_baseline_score = 0
    best_baseline_name = ""
    
    for name in baseline_names:
        results = all_results.get(name, [])
        if results and results[0]['final_score'] > best_baseline_score:
            best_baseline_score = results[0]['final_score']
            best_baseline_name = name
    
    your_results = all_results.get(your_method_name, [])
    your_score = your_results[0]['final_score'] if your_results else 0
    
    if best_baseline_score > 0:
        improvement_abs = your_score - best_baseline_score
        improvement_pct = (improvement_abs / best_baseline_score) * 100
        improvement_symbol = "↑" if improvement_abs > 0 else "↓"
    else:
        improvement_abs = 0
        improvement_pct = 0
        improvement_symbol = "?"
    
    improvement_rows = [
        ["Best Baseline", best_baseline_name, f"{best_baseline_score:.1f}%"],
        ["Your Method", your_method_name, f"{your_score:.1f}%"],
        ["Absolute Improvement", f"{improvement_symbol}{abs(improvement_abs):.1f}%", 
         f"{improvement_abs:+.1f} percentage points"],
        ["Relative Improvement", f"{improvement_symbol}{abs(improvement_pct):.1f}%", 
         f"{improvement_pct:+.1f}% better than baseline"]
    ]
    
    print(tabulate(improvement_rows, headers=["Metric", "Method", "Top-1 Score"], tablefmt="grid"))
    
    # ============================================================
    # VISUAL BAR CHART
    # ============================================================
    
    print("\n" + "="*90)
    print("📊 VISUAL COMPARISON: Top-1 Scores")
    print("="*90)
    
    # Find max score for scaling
    all_scores = []
    for results in all_results.values():
        if results:
            all_scores.append(results[0]['final_score'])
    max_score = max(all_scores) if all_scores else 100
    
    for method_name, results in all_results.items():
        if results:
            score = results[0]['final_score']
            bar_length = int((score / max_score) * 40) if max_score > 0 else 0
            bar = "█" * bar_length + "░" * (40 - bar_length)
            
            # Mark the best method
            is_best = (score == max_score)
            marker = " ← BEST" if is_best else ""
            
            # Mark your method
            is_yours = (method_name == your_method_name)
            yours_marker = " ★ YOUR METHOD" if is_yours else ""
            
            print(f"{method_name[:25]:<25} | {bar} | {score:5.1f}%{marker}{yours_marker}")
    
    # ============================================================
    # TABLE 4: Ranking Agreement (Jaccard Similarity)
    # ============================================================
    
    print("\n" + "="*90)
    print("🔄 TABLE 4: Ranking Agreement with Best Method")
    print("="*90)
    
    # Use best method as reference
    reference_name = your_method_name
    reference_results = all_results[reference_name]
    reference_job_ids = [r.get('job_id') for r in reference_results[:10] if 'job_id' in r]
    
    agreement_rows = [["Method", "Agreement (Jaccard)", "Overlap in Top-10", "Avg Score Difference"]]
    
    for method_name, results in all_results.items():
        if method_name != reference_name:
            method_job_ids = [r.get('job_id') for r in results[:10] if 'job_id' in r]
            
            # Calculate Jaccard similarity
            ref_set = set(reference_job_ids)
            method_set = set(method_job_ids)
            
            if ref_set and method_set:
                intersection = len(ref_set & method_set)
                union = len(ref_set | method_set)
                jaccard = (intersection / union) * 100 if union > 0 else 0
                overlap = f"{intersection}/10"
                
                # Calculate average score difference for overlapping jobs
                score_diffs = []
                for i, job_id in enumerate(method_job_ids[:10]):
                    if job_id in ref_set:
                        ref_idx = reference_job_ids.index(job_id)
                        method_idx = i
                        score_diff = abs(reference_results[ref_idx]['final_score'] - results[method_idx]['final_score'])
                        score_diffs.append(score_diff)
                avg_diff = np.mean(score_diffs) if score_diffs else 0
            else:
                jaccard = 0
                overlap = "0/10"
                avg_diff = 0
            
            agreement_rows.append([method_name, f"{jaccard:.1f}%", overlap, f"{avg_diff:.1f} points"])
    
    print(tabulate(agreement_rows, headers="firstrow", tablefmt="grid"))
    
    # ============================================================
    # TABLE 5: Rank Correlation
    # ============================================================
    
    print("\n" + "="*90)
    print("📈 TABLE 5: Rank Correlation with Best Method")
    print("="*90)
    
    correlation_rows = [["Method", "Avg Rank Difference", "Spearman (approx)", "NDCG@10"]]
    
    for method_name, results in all_results.items():
        if method_name != reference_name:
            rank_diffs = []
            ndcg_gains = []
            
            for i, r in enumerate(results[:10]):
                job_id = r.get('job_id')
                if job_id and job_id in reference_job_ids:
                    ref_rank = reference_job_ids.index(job_id)
                    rank_diff = abs(i - ref_rank)
                    rank_diffs.append(rank_diff)
                    
                    # Calculate NDCG-style gain
                    gain = 1 / (ref_rank + 1)  # Higher rank = higher gain
                    ndcg_gains.append(gain)
            
            avg_diff = np.mean(rank_diffs) if rank_diffs else 10
            # Approximate Spearman: lower avg diff = higher correlation
            approx_spearman = max(0, 1 - (avg_diff / 10))
            
            # Simplified NDCG (normalized by ideal)
            ideal_gains = [1/(i+1) for i in range(min(10, len(rank_diffs)))]
            ndcg = sum(ndcg_gains) / sum(ideal_gains) if ideal_gains else 0
            
            correlation_rows.append([method_name, f"{avg_diff:.1f} ranks", f"{approx_spearman:.3f}", f"{ndcg:.3f}"])
    
    print(tabulate(correlation_rows, headers="firstrow", tablefmt="grid"))
    
    # ============================================================
    # TABLE 6: Component Breakdown (if available)
    # ============================================================
    
    # Check if any results have breakdown info
    has_breakdown = False
    for results in all_results.values():
        if results and 'breakdown' in results[0]:
            has_breakdown = True
            break
    
    if has_breakdown:
        print("\n" + "="*90)
        print("🔬 TABLE 6: Component Score Breakdown (Top-1 Match)")
        print("="*90)
        
        breakdown_rows = [["Method", "Total Score", "Component 1", "Component 2", "Component 3"]]
        
        for method_name, results in all_results.items():
            if results and 'breakdown' in results[0]:
                total = results[0]['final_score']
                b = results[0]['breakdown']
                
                # Extract component names and values
                components = []
                for key, value in b.items():
                    if key not in ['method', 'cv_skills_count', 'job_skills_count', 'matched_skills', 
                                   'jaccard_score', 'skills_used', 'years_required']:
                        if isinstance(value, (int, float)) and value <= 100:
                            components.append(f"{key.replace('_match', '')}: {value:.1f}%")
                
                comp_str = " | ".join(components[:3])
                breakdown_rows.append([method_name, f"{total:.1f}%", comp_str, "", ""])
            else:
                breakdown_rows.append([method_name, f"{results[0]['final_score']:.1f}%" if results else "N/A", "No breakdown", "", ""])
        
        print(tabulate(breakdown_rows, headers="firstrow", tablefmt="grid"))
    
    # ============================================================
    # SUMMARY AND CONCLUSION
    # ============================================================
    
    print("\n" + "="*90)
    print("📝 SUMMARY & CONCLUSIONS")
    print("="*90)
    
    # Calculate improvements over each baseline
    print("\n📊 Performance Summary:")
    print("-" * 50)
    
    for baseline_name in baseline_names:
        baseline_results = all_results.get(baseline_name, [])
        if baseline_results:
            baseline_score = baseline_results[0]['final_score']
            improvement = ((your_score - baseline_score) / baseline_score) * 100 if baseline_score > 0 else 0
            
            if improvement > 20:
                verdict = "🎉 SIGNIFICANT improvement"
            elif improvement > 10:
                verdict = "👍 Good improvement"
            elif improvement > 0:
                verdict = "✅ Slight improvement"
            elif improvement > -10:
                verdict = "⚠️ Comparable performance"
            else:
                verdict = "❌ Worse than baseline"
            
            print(f"  vs {baseline_name}: {improvement:+.1f}% better → {verdict}")
    
    print("\n🎯 Key Findings:")
    print("-" * 50)
    
    if improvement_pct > 20:
        print("  ✅ Your method significantly outperforms all baselines")
        print("  ✅ The additional features (semantic, synonyms, seniority) provide real value")
        print("  ✅ Recommendation: Use your full method for production")
    elif improvement_pct > 10:
        print("  ✅ Your method shows clear improvement over baselines")
        print("  ✅ The added complexity is justified by performance gains")
    elif improvement_pct > 0:
        print("  ⚠️ Your method shows modest improvement over baselines")
        print("  ⚠️ Consider if the added complexity is worth the marginal gain")
    else:
        print("  ❌ Your method does not outperform baselines")
        print("  ❌ Reconsider features or simplify approach")
    
    # Calculate ranking quality
    avg_agreement = 0
    for baseline_name in baseline_names:
        baseline_results = all_results.get(baseline_name, [])
        if baseline_results:
            baseline_job_ids = [r.get('job_id') for r in baseline_results[:10] if 'job_id' in r]
            ref_set = set(reference_job_ids)
            method_set = set(baseline_job_ids)
            if ref_set and method_set:
                jaccard = len(ref_set & method_set) / len(ref_set | method_set)
                avg_agreement += jaccard
    
    if baseline_names:
        avg_agreement /= len(baseline_names)
        print(f"\n  📊 Average ranking agreement between methods: {avg_agreement*100:.1f}%")
        
        if avg_agreement < 0.3:
            print("  🔄 Methods produce very different rankings - feature choices matter greatly")
        elif avg_agreement < 0.6:
            print("  🔄 Moderate agreement - methods have different strengths")
        else:
            print("  🔄 High agreement - most methods converge on similar rankings")
    
    # ============================================================
    # SAVE COMBINED RESULTS
    # ============================================================
    
    output_path = "comparison_results.json"
    combined_results = {
        "methods": list(all_results.keys()),
        "top_scores": {name: results[0]['final_score'] if results else 0 for name, results in all_results.items()},
        "full_results": all_results,
        "improvement": {
            "best_baseline": best_baseline_name,
            "best_baseline_score": best_baseline_score,
            "your_method": your_method_name,
            "your_score": your_score,
            "absolute_improvement": improvement_abs,
            "relative_improvement_pct": improvement_pct
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Combined results saved to: {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare Results from Different Matchers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_all.py --files my_results.json tfidf_results.json
  
  # Compare with custom names
  python compare_all.py --files my_results.json tfidf_results.json jaccard_results.json --names "Our Method" "TF-IDF" "Jaccard"
  
  # Compare all baselines
  python compare_all.py --files my_results.json tfidf_results.json jaccard_results.json simple_results.json
        """
    )
    parser.add_argument("--files", nargs="+", required=True, 
                       help="JSON result files to compare (at least 2)")
    parser.add_argument("--names", nargs="+", 
                       help="Method names (optional, must match number of files)")
    parser.add_argument("--top", type=int, default=10, 
                       help="Number of top matches to consider (default: 10)")
    args = parser.parse_args()
    
    if args.names and len(args.names) != len(args.files):
        print("Error: Number of names must match number of files")
        return
    
    compare_all_results(args.files, args.names, args.top)


if __name__ == "__main__":
    main()