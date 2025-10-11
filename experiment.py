#!/usr/bin/env python3
"""
ExperimentResult and ExperimentSaver classes for MedAgents-2
"""
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from medagents import MedAgentsLog

@dataclass
class ExperimentResult:
    """Structured result for a single experiment run."""
    realidx: int
    question: str
    options: Dict[str, str]
    answer_idx: Optional[str] = None  # Ground truth answer
    difficulty: str = "auto"
    time_taken: float = None
    medagents_log: Dict[str, Any] = None  # Store as dict instead of object
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization, merging logs directly."""
        if not self.medagents_log:
            return {
                'realidx': self.realidx,
                'question': self.question,
                'options': self.options,
                'answer_idx': self.answer_idx,
                'difficulty': self.difficulty,
                'time_taken': self.time_taken
            }
        result_dict = {
            'realidx': self.realidx,
            'question': self.question,
            'options': self.options,
            'answer_idx': self.answer_idx,
            'difficulty': self.difficulty,
            'time_taken': self.time_taken,
            'medagents_log': self.medagents_log
        }
        result_dict.update(self.medagents_log)
        return result_dict
    
    @classmethod
    def from_medagents_log(cls, realidx: int, question: str, options: Dict[str, str], 
                          answer_idx: Optional[str], difficulty: str, time_taken: float, medagents_log: MedAgentsLog) -> 'ExperimentResult':
        """Create ExperimentResult from MedAgentsLog object."""
        log_dict = medagents_log.to_dict()
        return cls(
            realidx=realidx,
            question=question,
            options=options,
            answer_idx=answer_idx,
            difficulty=difficulty,
            time_taken=time_taken,
            medagents_log=log_dict
        )

class ExperimentSaver:
    """Handles saving experiment results with different detail levels to separate files."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.summary_file = os.path.join(output_dir, "summary.json")
        self.logs_file = os.path.join(output_dir, "logs.json")
        self.usage_file = os.path.join(output_dir, "usage.json")
        self.metrics_file = os.path.join(output_dir, "metrics.json")
        self.results: List[ExperimentResult] = []
        self._load_existing_results()
    def _load_existing_results(self):
        """Load existing results if available."""
        if os.path.exists(self.summary_file):
            print(f"Loading existing results from: {self.summary_file}")
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    basic_fields = {
                        'realidx': item['realidx'],
                        'question': item['question'],
                        'options': item['options'],
                        'answer_idx': item.get('answer_idx'),
                        'difficulty': item['difficulty'],
                        'time_taken': item['time_taken'],
                    }
                    medagents_log = {}
                    if os.path.exists(self.logs_file):
                        with open(self.logs_file, 'r', encoding='utf-8') as logs_f:
                            logs_data = json.load(logs_f)
                            if str(item['realidx']) in logs_data:
                                medagents_log = logs_data[str(item['realidx'])]
                    result = ExperimentResult(
                        medagents_log=medagents_log,
                        **basic_fields
                    )
                    self.results.append(result)
            print(f"Loaded {len(self.results)} existing results.")
    def add_result(self, result: ExperimentResult):
        """Add a new result and save immediately."""
        self.results.append(result)
        self.save_results()
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics for the experiment."""
        sorted_results = sorted(self.results, key=lambda x: x.realidx)
        correct_answers = 0
        total_answers = 0
        difficulty_stats = {}
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        requests = 0
        cached_tokens = 0
        reasoning_tokens = 0
        total_time = 0
        valid_time_count = 0
        for result in sorted_results:
            if result.answer_idx and result.medagents_log and 'final_decision' in result.medagents_log:
                total_answers += 1
                predicted_answer = result.medagents_log['final_decision'].get('final_answer')
                is_correct = predicted_answer == result.answer_idx
                if is_correct:
                    correct_answers += 1
                difficulty = result.difficulty
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {'correct': 0, 'total': 0}
                difficulty_stats[difficulty]['total'] += 1
                if is_correct:
                    difficulty_stats[difficulty]['correct'] += 1
            if result.medagents_log and 'total_usage' in result.medagents_log:
                usage = result.medagents_log['total_usage']
                total_tokens += usage.get('total_tokens', 0)
                input_tokens += usage.get('input_tokens', 0)
                output_tokens += usage.get('output_tokens', 0)
                requests += usage.get('requests', 0)
                cached_tokens += usage.get('input_tokens_details', {}).get('cached_tokens', 0)
                reasoning_tokens += usage.get('output_tokens_details', {}).get('reasoning_tokens', 0)
            if result.time_taken is not None:
                total_time += result.time_taken
                valid_time_count += 1
        avg_time = total_time / valid_time_count if valid_time_count > 0 else 0
        avg_total_tokens = total_tokens / len(sorted_results) if sorted_results else 0
        avg_input_tokens = input_tokens / len(sorted_results) if sorted_results else 0
        avg_output_tokens = output_tokens / len(sorted_results) if sorted_results else 0
        avg_requests = requests / len(sorted_results) if sorted_results else 0
        difficulty_accuracy = {}
        for difficulty, stats in difficulty_stats.items():
            difficulty_accuracy[difficulty] = {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total']
            }
        overall_accuracy = correct_answers / total_answers if total_answers > 0 else 0
        return {
            'accuracy_metrics': {
                'overall_accuracy': overall_accuracy,
                'correct_answers': correct_answers,
                'total_answers': total_answers,
                'difficulty_breakdown': difficulty_accuracy
            },
            'token_usage_metrics': {
                'total_usage': {
                    'total_tokens': total_tokens,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'requests': requests,
                    'cached_tokens': cached_tokens,
                    'reasoning_tokens': reasoning_tokens
                },
                'average_usage': {
                    'avg_total_tokens': avg_total_tokens,
                    'avg_input_tokens': avg_input_tokens,
                    'avg_output_tokens': avg_output_tokens,
                    'avg_requests': avg_requests,
                    'avg_cached_tokens': cached_tokens / len(sorted_results) if sorted_results else 0,
                    'avg_reasoning_tokens': reasoning_tokens / len(sorted_results) if sorted_results else 0
                }
            },
            'time_metrics': {
                'total_time': total_time,
                'average_time': avg_time,
                'valid_time_count': valid_time_count,
                'total_problems': len(sorted_results)
            },
            'experiment_summary': {
                'total_problems_processed': len(sorted_results),
                'problems_with_answers': total_answers,
                'problems_with_valid_time': valid_time_count,
                'timestamp': datetime.now().isoformat()
            }
        }
    def save_results(self):
        """Save results to different files."""
        sorted_results = sorted(self.results, key=lambda x: x.realidx)
        summary_data = []
        for result in sorted_results:
            summary_item = {
                'realidx': result.realidx,
                'question': result.question,
                'options': result.options,
                'answer_idx': result.answer_idx,
                'difficulty': result.difficulty,
                'time_taken': result.time_taken,
            }
            if result.medagents_log and 'final_decision' in result.medagents_log:
                summary_item.update({
                    'final_answer': result.medagents_log['final_decision'].get('final_answer'),
                    'vote_scores': result.medagents_log['final_decision'].get('vote_scores'),
                    'expert_details': result.medagents_log['final_decision'].get('details'),
                })
            summary_data.append(summary_item)
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logs_data = {}
        for result in sorted_results:
            if result.medagents_log:
                logs_data[str(result.realidx)] = result.medagents_log
        with open(self.logs_file, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False)
        usage_data = {}
        total_usage = {
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'requests': 0,
            'cached_tokens': 0,
            'reasoning_tokens': 0
        }
        for result in sorted_results:
            if result.medagents_log and 'total_usage' in result.medagents_log:
                usage = result.medagents_log['total_usage']
                usage_data[str(result.realidx)] = usage
                total_usage['total_tokens'] += usage.get('total_tokens', 0)
                total_usage['input_tokens'] += usage.get('input_tokens', 0)
                total_usage['output_tokens'] += usage.get('output_tokens', 0)
                total_usage['requests'] += usage.get('requests', 0)
                total_usage['cached_tokens'] += usage.get('input_tokens_details', {}).get('cached_tokens', 0)
                total_usage['reasoning_tokens'] += usage.get('output_tokens_details', {}).get('reasoning_tokens', 0)
        usage_data['total'] = total_usage
        correct_answers = 0
        total_answers = 0
        for result in sorted_results:
            if result.answer_idx and result.medagents_log and 'final_decision' in result.medagents_log:
                total_answers += 1
                if result.medagents_log['final_decision'].get('final_answer') == result.answer_idx:
                    correct_answers += 1
        accuracy = correct_answers / total_answers if total_answers > 0 else 0
        usage_data['accuracy'] = {
            'correct_answers': correct_answers,
            'total_answers': total_answers,
            'accuracy_rate': accuracy
        }
        with open(self.usage_file, 'w', encoding='utf-8') as f:
            json.dump(usage_data, f, indent=2, ensure_ascii=False)
        metrics = self._calculate_metrics()
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    def get_processed_realidx(self) -> set:
        """Get set of already processed realidx values."""
        return {result.realidx for result in self.results} 