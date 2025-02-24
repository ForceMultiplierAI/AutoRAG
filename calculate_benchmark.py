import json
import argparse
from collections import defaultdict

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calculate_accuracy(results):
    if not results:
        return 0
    return sum(1 for r in results if r) / len(results) * 100

def print_results_table(stats):
    print("\nLongBench v2 Evaluation Results")
    print("=" * 80)
    print(f"{'Category':<20} {'Score':<10} {'Count':<10}")
    print("-" * 80)
    
    # Overall scores
    print(f"{'Overall':<20} {stats['overall_acc']:.1f}% {stats['total_count']}")
    
    # Difficulty breakdown
    print("\nBy Difficulty:")
    for diff in ['easy', 'hard']:
        count = len(stats['difficulty'][diff])
        acc = calculate_accuracy(stats['difficulty'][diff])
        print(f"{diff.capitalize():<20} {acc:.1f}% {count}")
    
    # Length breakdown
    print("\nBy Length:")
    for length in ['short', 'medium', 'long']:
        count = len(stats['length'][length])
        acc = calculate_accuracy(stats['length'][length])
        print(f"{length.capitalize():<20} {acc:.1f}% {count}")
    
    # Domain breakdown
    print("\nBy Domain:")
    for domain in sorted(stats['domain'].keys()):
        count = len(stats['domain'][domain])
        acc = calculate_accuracy(stats['domain'][domain])
        print(f"{domain[:19]:<20} {acc:.1f}% {count}")

def main(input_file):
    # Load data
    data = load_jsonl(input_file)
    
    # Initialize statistics
    stats = {
        'total_count': len(data),
        'correct_count': sum(1 for item in data if item['is_correct']),
        'difficulty': defaultdict(list),
        'length': defaultdict(list),
        'domain': defaultdict(list),
        'subdomain': defaultdict(list)
    }
    
    # Calculate overall accuracy
    stats['overall_acc'] = (stats['correct_count'] / stats['total_count'] * 100) if stats['total_count'] > 0 else 0
    
    # Collect results by category
    for item in data:
        stats['difficulty'][item['difficulty']].append(item['is_correct'])
        stats['length'][item['length']].append(item['is_correct'])
        stats['domain'][item['domain']].append(item['is_correct'])
        stats['subdomain'][item['sub_domain']].append(item['is_correct'])
    
    # Print results
    print_results_table(stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate LongBench v2 benchmark scores')
    parser.add_argument('--input', type=str, default='output.jsonl',
                        help='Input JSONL file (default: output.jsonl)')
    args = parser.parse_args()
    main(args.input)
