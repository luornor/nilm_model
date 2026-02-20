"""
Assess transfer learning potential between source and target devices.

This script analyzes power signatures to identify which source models
are best suited for transfer learning to new target devices.

Usage:
    python scripts/assess_transfer_learning_potential.py \
        --source-data Exports/lit_synth_5s_states.csv \
        --target-data Exports/dataset1_1s.csv \
        --output outputs/transfer_learning_assessment.json

Author: NILM Framework
Date: 2026-02-13
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def extract_power_signature(df: pd.DataFrame, appliance_col: str) -> Dict:
    """
    Extract power signature statistics for an appliance.
    
    Args:
        df: DataFrame with power and appliance state
        appliance_col: Column name (e.g., 'y_Laptop')
    
    Returns:
        Dictionary with signature features
    """
    on_mask = df[appliance_col] == 1
    off_mask = df[appliance_col] == 0
    
    if not on_mask.any():
        return None
    
    # Power statistics
    power_on = df.loc[on_mask, 'P'].values
    power_off = df.loc[off_mask, 'P'].values if off_mask.any() else np.array([0])
    
    # Transition analysis
    transitions = df[appliance_col].diff().fillna(0)
    on_transitions = df.loc[transitions == 1, 'P'].values
    off_transitions = df.loc[transitions == -1, 'P'].values
    
    signature = {
        'appliance': appliance_col[2:],  # Remove 'y_' prefix
        'power_stats': {
            'mean_on': float(np.mean(power_on)),
            'std_on': float(np.std(power_on)),
            'median_on': float(np.median(power_on)),
            'mean_off': float(np.mean(power_off)),
            'std_off': float(np.std(power_off)),
            'delta': float(np.mean(power_on) - np.mean(power_off)),
        },
        'on_rate': float(on_mask.mean()),
        'num_events': float(np.abs(transitions).sum() / 2),
        'avg_on_duration': float(on_mask.sum() / (np.abs(transitions).sum() / 2)) if transitions.sum() > 0 else 0,
        'transition_stats': {
            'mean_on_step': float(np.mean(on_transitions)) if len(on_transitions) > 0 else 0,
            'std_on_step': float(np.std(on_transitions)) if len(on_transitions) > 0 else 0,
            'mean_off_step': float(np.mean(off_transitions)) if len(off_transitions) > 0 else 0,
            'std_off_step': float(np.std(off_transitions)) if len(off_transitions) > 0 else 0,
        },
        'distribution': {
            'skewness': float(stats.skew(power_on)),
            'kurtosis': float(stats.kurtosis(power_on)),
        }
    }
    
    return signature


def compute_signature_similarity(sig1: Dict, sig2: Dict) -> float:
    """
    Compute similarity score between two power signatures.
    
    Higher score = more similar = better transfer learning candidate
    
    Score components:
    1. Power level similarity (delta)
    2. Statistical similarity (std, skewness, kurtosis)
    3. Temporal similarity (on_rate, avg_on_duration)
    4. Transition similarity (on/off steps)
    
    Returns:
        Similarity score in [0, 1]
    """
    if sig1 is None or sig2 is None:
        return 0.0
    
    # Power level similarity (normalized by max delta)
    delta1 = abs(sig1['power_stats']['delta'])
    delta2 = abs(sig2['power_stats']['delta'])
    max_delta = max(delta1, delta2) if max(delta1, delta2) > 0 else 1
    power_sim = 1.0 - abs(delta1 - delta2) / max_delta
    
    # Statistical distribution similarity
    skew1, skew2 = sig1['distribution']['skewness'], sig2['distribution']['skewness']
    kurt1, kurt2 = sig1['distribution']['kurtosis'], sig2['distribution']['kurtosis']
    
    skew_sim = 1.0 - min(abs(skew1 - skew2) / 10.0, 1.0)  # Normalize by typical range
    kurt_sim = 1.0 - min(abs(kurt1 - kurt2) / 20.0, 1.0)
    
    # Temporal pattern similarity
    on_rate1, on_rate2 = sig1['on_rate'], sig2['on_rate']
    on_rate_sim = 1.0 - abs(on_rate1 - on_rate2)
    
    duration1 = sig1['avg_on_duration']
    duration2 = sig2['avg_on_duration']
    max_duration = max(duration1, duration2) if max(duration1, duration2) > 0 else 1
    duration_sim = 1.0 - abs(duration1 - duration2) / max_duration
    
    # Transition similarity
    trans1 = sig1['transition_stats']['mean_on_step']
    trans2 = sig2['transition_stats']['mean_on_step']
    max_trans = max(trans1, trans2) if max(trans1, trans2) > 0 else 1
    trans_sim = 1.0 - abs(trans1 - trans2) / max_trans
    
    # Weighted combination
    weights = {
        'power': 0.35,      # Most important
        'skewness': 0.10,
        'kurtosis': 0.10,
        'on_rate': 0.15,
        'duration': 0.15,
        'transition': 0.15,
    }
    
    similarity = (
        weights['power'] * power_sim +
        weights['skewness'] * skew_sim +
        weights['kurtosis'] * kurt_sim +
        weights['on_rate'] * on_rate_sim +
        weights['duration'] * duration_sim +
        weights['transition'] * trans_sim
    )
    
    return similarity


def find_best_transfer_candidates(source_signatures: Dict[str, Dict],
                                   target_signatures: Dict[str, Dict],
                                   top_k: int = 3) -> Dict:
    """
    Find best source models for each target device.
    
    Args:
        source_signatures: Dict mapping source appliance -> signature
        target_signatures: Dict mapping target appliance -> signature
        top_k: Number of top candidates to return
    
    Returns:
        Dict mapping target appliance -> list of (source, similarity) tuples
    """
    recommendations = {}
    
    for target_name, target_sig in target_signatures.items():
        if target_sig is None:
            continue
        
        similarities = []
        for source_name, source_sig in source_signatures.items():
            if source_sig is None:
                continue
            
            sim = compute_signature_similarity(source_sig, target_sig)
            similarities.append((source_name, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        recommendations[target_name] = similarities[:top_k]
    
    return recommendations


def classify_load_type(signature: Dict) -> str:
    """
    Infer load type from power signature.
    
    Heuristics:
    - Resistive: Low std_on, low skewness, stable power
    - Reactive: Higher std_on, possible oscillations
    - Switched-source: High std_on, high skewness (spiky)
    
    Returns:
        'resistive', 'reactive', or 'switched_source'
    """
    if signature is None:
        return 'unknown'
    
    std_on = signature['power_stats']['std_on']
    mean_on = signature['power_stats']['mean_on']
    skewness = abs(signature['distribution']['skewness'])
    
    # Coefficient of variation
    cv = std_on / mean_on if mean_on > 0 else 0
    
    # Classification rules
    if cv < 0.15 and skewness < 0.5:
        return 'resistive'
    elif cv > 0.3 or skewness > 1.0:
        return 'switched_source'
    else:
        return 'reactive'


def generate_transfer_learning_plan(recommendations: Dict,
                                    source_signatures: Dict,
                                    target_signatures: Dict) -> Dict:
    """
    Generate actionable transfer learning plan.
    
    Returns:
        Dictionary with recommendations and commands
    """
    plan = {
        'summary': {},
        'recommendations': [],
        'training_commands': [],
    }
    
    for target, candidates in recommendations.items():
        if not candidates:
            continue
        
        best_source, best_sim = candidates[0]
        
        # Classify as good/moderate/poor transfer candidate
        if best_sim > 0.7:
            quality = 'excellent'
        elif best_sim > 0.5:
            quality = 'good'
        elif best_sim > 0.3:
            quality = 'moderate'
        else:
            quality = 'poor'
        
        target_sig = target_signatures.get(target)
        source_sig = source_signatures.get(best_source)
        
        plan['summary'][target] = {
            'best_source': best_source,
            'similarity': float(best_sim),
            'quality': quality,
            'target_load_type': classify_load_type(target_sig),
            'source_load_type': classify_load_type(source_sig),
            'power_delta_ratio': float(
                target_sig['power_stats']['delta'] / source_sig['power_stats']['delta']
            ) if source_sig and source_sig['power_stats']['delta'] > 0 else 1.0,
        }
        
        # Generate recommendation text
        rec_text = (
            f"{target}:\n"
            f"  Best source: {best_source} (similarity: {best_sim:.3f}, {quality})\n"
            f"  Target: {target_sig['power_stats']['delta']:.1f}W delta, "
            f"{target_sig['on_rate']*100:.1f}% ON rate\n"
            f"  Source: {source_sig['power_stats']['delta']:.1f}W delta, "
            f"{source_sig['on_rate']*100:.1f}% ON rate"
        )
        
        if best_sim > 0.5:
            rec_text += "\n  ✓ Recommended for transfer learning"
        elif best_sim > 0.3:
            rec_text += "\n  ⚠ Moderate transfer potential, consider fine-tuning"
        else:
            rec_text += "\n  ✗ Poor match, train from scratch recommended"
        
        plan['recommendations'].append(rec_text)
        
        # Generate training command
        if best_sim > 0.3:
            cmd = (
                f"# Transfer learning: {best_source} -> {target}\n"
                f"python scripts/transfer_learning.py \\\n"
                f"  --source-model outputs/training/models/cnn_seq2point_y_{best_source}.pt \\\n"
                f"  --target-data <target_dataset.csv> \\\n"
                f"  --target-appliance y_{target} \\\n"
                f"  --output outputs/training/models/transferred/cnn_seq2point_y_{target}_transferred.pt \\\n"
                f"  --epochs 15 \\\n"
                f"  --learning-rate 0.0001"
            )
            plan['training_commands'].append(cmd)
    
    return plan


def main():
    parser = argparse.ArgumentParser(
        description='Assess transfer learning potential for new devices'
    )
    parser.add_argument('--source-data', type=str, required=True,
                        help='Source dataset CSV (trained models)')
    parser.add_argument('--target-data', type=str, required=True,
                        help='Target dataset CSV (new devices)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file for recommendations')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of top candidates per target device')
    
    args = parser.parse_args()
    
    # Load datasets
    print(f"Loading source data: {args.source_data}")
    source_df = pd.read_csv(args.source_data)
    
    print(f"Loading target data: {args.target_data}")
    target_df = pd.read_csv(args.target_data)
    
    # Extract signatures
    print("\nExtracting power signatures...")
    
    source_appliances = [col for col in source_df.columns if col.startswith('y_')]
    target_appliances = [col for col in target_df.columns if col.startswith('y_')]
    
    print(f"Source appliances: {len(source_appliances)}")
    print(f"Target appliances: {len(target_appliances)}")
    
    source_signatures = {}
    for col in source_appliances:
        sig = extract_power_signature(source_df, col)
        if sig:
            source_signatures[sig['appliance']] = sig
    
    target_signatures = {}
    for col in target_appliances:
        sig = extract_power_signature(target_df, col)
        if sig:
            target_signatures[sig['appliance']] = sig
    
    # Find transfer learning candidates
    print("\nComputing similarity scores...")
    recommendations = find_best_transfer_candidates(
        source_signatures, target_signatures, args.top_k
    )
    
    # Generate plan
    print("\nGenerating transfer learning plan...")
    plan = generate_transfer_learning_plan(
        recommendations, source_signatures, target_signatures
    )
    
    # Print recommendations
    print("\n" + "="*70)
    print("TRANSFER LEARNING RECOMMENDATIONS")
    print("="*70)
    
    for rec in plan['recommendations']:
        print(rec)
        print()
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'source_signatures': source_signatures,
            'target_signatures': target_signatures,
            'recommendations': recommendations,
            'plan': plan,
        }, f, indent=2)
    
    print(f"\nFull results saved to: {output_path}")
    
    # Print training commands
    if plan['training_commands']:
        print("\n" + "="*70)
        print("TRAINING COMMANDS")
        print("="*70)
        for cmd in plan['training_commands']:
            print(cmd)
            print()


if __name__ == '__main__':
    main()
