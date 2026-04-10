import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# this is to extract answer in \boxed{}
def extract_gsm8k_answer(text: str) -> Optional[str]:
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


# to run python
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)


def compare_tool_calls(gold: str, pred: str) -> bool:
    """
    Robust comparison for tool calls (similar to BFCL AST matching).
    """
    import ast
    import re

    def normalize_val(val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Try to evaluate simple math expressions like 1/6
            if re.match(r'^[\d\s\/\.\-\+\*]+$', val):
                try:
                    return float(eval(val))
                except: pass
            return val.strip().replace("'", '"')
        return val

    def parse_call(call_str):
        call_str = call_str.strip()
        if not call_str or call_str == "None":
            return None, {}
        
        # Handle cases where pred might have extra text but we already extracted it partially
        # If it doesn't look like a call, just return it as name
        if '(' not in call_str:
            return call_str, {}
        
        try:
            tree = ast.parse(call_str)
            if not tree.body or not isinstance(tree.body[0], ast.Expr) or not isinstance(tree.body[0].value, ast.Call):
                return None, {}
            
            call = tree.body[0].value
            func_name = ast.unparse(call.func).strip()
            
            args = {}
            for kw in call.keywords:
                val = ast.unparse(kw.value).strip()
                # Try to get constant value if possible
                try:
                    const_val = ast.literal_eval(val)
                    args[kw.arg] = normalize_val(const_val)
                except:
                    args[kw.arg] = normalize_val(val)
            
            pos_args = []
            for a in call.args:
                val = ast.unparse(a).strip()
                try:
                    const_val = ast.literal_eval(val)
                    pos_args.append(normalize_val(const_val))
                except:
                    pos_args.append(normalize_val(val))
            
            return func_name, {'kwargs': args, 'pos_args': pos_args}
        except:
            return None, {}

    gold_name, gold_data = parse_call(gold)
    pred_name, pred_data = parse_call(pred)
    
    if not gold_name or not pred_name:
        # Fallback to fuzzy string match if AST fails
        g_clean = re.sub(r'\s+', '', gold.lower())
        p_clean = re.sub(r'\s+', '', pred.lower())
        return g_clean == p_clean
        
    if gold_name != pred_name:
        return False
        
    # Compare positional args
    if len(gold_data.get('pos_args', [])) != len(pred_data.get('pos_args', [])):
        return False
    for g_a, p_a in zip(gold_data['pos_args'], pred_data['pos_args']):
        if g_a != p_a:
            # Try float comparison for strings that might be numbers
            try:
                if abs(float(g_a) - float(p_a)) < 1e-6: continue
            except: pass
            return False

    # Compare keyword args
    g_kwargs = gold_data.get('kwargs', {})
    p_kwargs = pred_data.get('kwargs', {})
    
    if len(g_kwargs) != len(p_kwargs):
        return False
        
    for k, g_v in g_kwargs.items():
        if k not in p_kwargs:
            return False
        p_v = p_kwargs[k]
        if g_v != p_v:
            # Try float comparison
            try:
                if abs(float(g_v) - float(p_v)) < 1e-6: continue
            except: pass
            return False
            
    return True
