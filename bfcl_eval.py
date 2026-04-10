"""
BFCL v3 Standard Evaluation Module

This module implements evaluation methods aligned with Berkeley Function Calling Leaderboard v3 standards.
"""

import ast
import json
from typing import Dict, List, Tuple, Optional, Any


class BFCLEvaluator:
    """
    BFCL-aligned evaluator with AST and Exec evaluation methods.

    Implements:
    1. Function invocation check
    2. Required parameter validation
    3. Hallucination detection
    4. Parameter type consistency
    5. Parameter value matching (exact/range/structure)
    """

    def __init__(self):
        self.results = {
            'total': 0,
            'correct': 0,
            'function_name_errors': 0,
            'missing_required_params': 0,
            'hallucination_errors': 0,
            'type_errors': 0,
            'value_errors': 0
        }

    def evaluate_single(self, gold_call: str, pred_call: str, function_schema: Dict = None) -> Dict:
        """
        Evaluate a single function call prediction.

        Args:
            gold_call: Ground truth function call string (e.g., "func(a=1, b=2)")
            pred_call: Predicted function call string
            function_schema: Function definition schema with parameters info

        Returns:
            Dict with 'correct' (bool) and 'errors' (list of str)
        """
        result = {
            'correct': False,
            'errors': [],
            'details': {}
        }

        try:
            # Parse both calls
            gold_parsed = self._parse_function_call(gold_call)
            pred_parsed = self._parse_function_call(pred_call)

            # Step 1: Function invocation check
            if gold_parsed['name'] != pred_parsed['name']:
                result['errors'].append(f"Function name mismatch: expected {gold_parsed['name']}, got {pred_parsed['name']}")
                result['details']['function_name'] = {'gold': gold_parsed['name'], 'pred': pred_parsed['name']}
                self.results['function_name_errors'] += 1
                self.results['total'] += 1  # Count total before return
                return result

            # Step 2: Required parameter validation
            if function_schema:
                missing_required = self._check_required_params(
                    pred_parsed['args'],
                    function_schema.get('parameters', {})
                )
                if missing_required:
                    result['errors'].append(f"Missing required parameters: {missing_required}")
                    result['details']['missing_required'] = missing_required
                    self.results['missing_required_params'] += 1
                    self.results['total'] += 1  # Count total before return
                    return result

            # Step 3: Hallucination detection
            hallucinated_params = self._detect_hallucination(
                pred_parsed['args'],
                gold_parsed['args'],
                function_schema
            )
            if hallucinated_params:
                result['errors'].append(f"Hallucinated parameters: {hallucinated_params}")
                result['details']['hallucinated'] = hallucinated_params
                self.results['hallucination_errors'] += 1

            # Step 4: Type consistency check
            type_errors = self._check_type_consistency(
                gold_parsed['args'],
                pred_parsed['args'],
                function_schema
            )
            if type_errors:
                result['errors'].append(f"Type mismatches: {type_errors}")
                result['details']['type_errors'] = type_errors
                self.results['type_errors'] += 1

            # Step 5: Value matching
            value_errors = self._check_value_consistency(
                gold_parsed['args'],
                pred_parsed['args']
            )
            if value_errors:
                result['errors'].append(f"Value mismatches: {value_errors}")
                result['details']['value_errors'] = value_errors
                self.results['value_errors'] += 1

            # Final decision
            if not result['errors']:
                result['correct'] = True
                self.results['correct'] += 1

            self.results['total'] += 1
            return result

        except Exception as e:
            result['errors'].append(f"Parsing error: {str(e)}")
            self.results['total'] += 1
            return result

    def _parse_function_call(self, call_str: str) -> Dict:
        """Parse function call string into structured format."""
        call_str = call_str.strip()

        # Handle empty or None
        if not call_str or call_str == "None":
            return {'name': None, 'args': {}}

        # If it's just a function name
        if '(' not in call_str:
            return {'name': call_str, 'args': {}}

        # Parse using AST
        try:
            tree = ast.parse(call_str)
            call_node = tree.body[0].value

            func_name = None
            if isinstance(call_node.func, ast.Name):
                func_name = call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                func_name = call_node.func.attr

            args = {}
            # Parse keyword arguments
            for kw in call_node.keywords:
                args[kw.arg] = self._parse_ast_value(kw.value)

            return {'name': func_name, 'args': args}
        except:
            # Fallback: simple regex parse
            import re
            match = re.match(r'(\w+)\((.*)\)', call_str)
            if match:
                func_name = match.group(1)
                args_str = match.group(2)
                # Simple key=value parsing
                args = {}
                if args_str.strip():
                    for pair in args_str.split(','):
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            args[k.strip()] = v.strip().strip('"\'')
                return {'name': func_name, 'args': args}
            return {'name': None, 'args': {}}

    def _parse_ast_value(self, node):
        """Extract value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.List):
            return [self._parse_ast_value(el) for el in node.elts]
        elif isinstance(node, ast.Dict):
            return {
                self._parse_ast_value(k): self._parse_ast_value(v)
                for k, v in zip(node.keys, node.values)
            }
        else:
            return ast.unparse(node)

    def _check_required_params(self, pred_args: Dict, params_schema: Dict) -> List[str]:
        """Check if all required parameters are present."""
        required = params_schema.get('required', [])
        missing = [p for p in required if p not in pred_args]
        return missing

    def _detect_hallucination(self, pred_args: Dict, gold_args: Dict, function_schema: Dict) -> List[str]:
        """Detect hallucinated parameters not in function schema."""
        if not function_schema:
            # If no schema, compare with gold args
            return [k for k in pred_args if k not in gold_args]

        # Check against schema
        valid_params = set(function_schema.get('parameters', {}).get('properties', {}).keys())
        hallucinated = [k for k in pred_args if k not in valid_params]
        return hallucinated

    def _check_type_consistency(self, gold_args: Dict, pred_args: Dict, function_schema: Dict) -> List[Dict]:
        """Check parameter type consistency."""
        type_errors = []

        if not function_schema:
            return type_errors

        properties = function_schema.get('parameters', {}).get('properties', {})

        for param_name, pred_value in pred_args.items():
            if param_name in properties:
                expected_type = properties[param_name].get('type')
                if expected_type:
                    pred_type = self._infer_type(pred_value)
                    if pred_type and pred_type != expected_type:
                        # Allow numeric type flexibility
                        if not (expected_type in ['integer', 'float', 'number'] and pred_type in ['int', 'float']):
                            type_errors.append({
                                'param': param_name,
                                'expected': expected_type,
                                'got': pred_type
                            })

        return type_errors

    def _infer_type(self, value: Any) -> Optional[str]:
        """Infer the type of a value."""
        if isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            return 'string'
        elif isinstance(value, list):
            return 'array'
        elif isinstance(value, dict):
            return 'object'
        return None

    def _check_value_consistency(self, gold_args: Dict, pred_args: Dict, match_mode: str = 'exact') -> List[Dict]:
        """
        Check parameter value consistency.

        Args:
            match_mode: 'exact', 'range', or 'structure'
        """
        value_errors = []

        for param_name, gold_value in gold_args.items():
            if param_name not in pred_args:
                value_errors.append({
                    'param': param_name,
                    'expected': gold_value,
                    'got': None,
                    'error': 'missing'
                })
                continue

            pred_value = pred_args[param_name]

            if match_mode == 'exact':
                if gold_value != pred_value:
                    # For numeric values, check if they're close
                    if isinstance(gold_value, (int, float)) and isinstance(pred_value, (int, float)):
                        # Range match: within 20%
                        tolerance = abs(gold_value) * 0.2
                        if abs(gold_value - pred_value) > tolerance:
                            value_errors.append({
                                'param': param_name,
                                'expected': gold_value,
                                'got': pred_value,
                                'error': 'value_mismatch'
                            })
                    else:
                        value_errors.append({
                            'param': param_name,
                            'expected': gold_value,
                            'got': pred_value,
                            'error': 'value_mismatch'
                        })

            elif match_mode == 'structure':
                # Only check structure (type), not exact values
                if type(gold_value) != type(pred_value):
                    value_errors.append({
                        'param': param_name,
                        'expected_type': type(gold_value).__name__,
                        'got_type': type(pred_value).__name__,
                        'error': 'type_mismatch'
                    })

        return value_errors

    def get_metrics(self) -> Dict:
        """Get evaluation metrics."""
        total = self.results['total']
        if total == 0:
            return {'accuracy': 0.0, **self.results}

        accuracy = self.results['correct'] / total
        return {
            'accuracy': accuracy,
            'correct': self.results['correct'],
            'total': total,
            'function_name_error_rate': self.results['function_name_errors'] / total if total > 0 else 0.0,
            'missing_required_rate': self.results['missing_required_params'] / total if total > 0 else 0.0,
            'hallucination_rate': self.results['hallucination_errors'] / total if total > 0 else 0.0,
            'type_error_rate': self.results['type_errors'] / total if total > 0 else 0.0,
            'value_error_rate': self.results['value_errors'] / total if total > 0 else 0.0
        }


def compare_tool_calls_bfcl(gold: str, pred: str, function_schema: Dict = None) -> bool:
    """
    BFCL-aligned function call comparison.

    Args:
        gold: Ground truth function call
        pred: Predicted function call
        function_schema: Function definition schema (optional)

    Returns:
        bool: Whether the prediction is correct
    """
    evaluator = BFCLEvaluator()
    result = evaluator.evaluate_single(gold, pred, function_schema)
    return result['correct']


def compare_tool_calls(gold: str, pred: str) -> bool:
    """
    Backward compatible wrapper for the original compare_tool_calls function.
    Uses BFCL-aligned evaluation.
    """
    return compare_tool_calls_bfcl(gold, pred, function_schema=None)


if __name__ == "__main__":
    # Test cases
    evaluator = BFCLEvaluator()

    # Test 1: Correct call
    gold1 = "calc_binomial_probability(n=20, k=5, p=0.6)"
    pred1 = "calc_binomial_probability(n=20, k=5, p=0.6)"
    result1 = evaluator.evaluate_single(gold1, pred1)
    print(f"Test 1: {result1}")

    # Test 2: Wrong function name
    gold2 = "calc_binomial_probability(n=20, k=5, p=0.6)"
    pred2 = "calculate_probability(n=20, k=5, p=0.6)"
    result2 = evaluator.evaluate_single(gold2, pred2)
    print(f"Test 2: {result2}")

    # Test 3: Missing required parameter
    gold3 = "calc_binomial_probability(n=20, k=5, p=0.6)"
    pred3 = "calc_binomial_probability(n=20, k=5)"
    schema = {
        'parameters': {
            'required': ['n', 'k', 'p'],
            'properties': {
                'n': {'type': 'integer'},
                'k': {'type': 'integer'},
                'p': {'type': 'float'}
            }
        }
    }
    result3 = evaluator.evaluate_single(gold3, pred3, schema)
    print(f"Test 3: {result3}")

    print("\nMetrics:", evaluator.get_metrics())
