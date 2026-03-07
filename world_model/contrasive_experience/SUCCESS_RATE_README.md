# Success Rate Computation

This document explains how success rates are computed and displayed in the CoMEM-Agent evaluation system.

## Automatic Display (Recommended)

Starting from the latest updates, **success rates are automatically computed and displayed** at the end of each evaluation run using `./run_agent.sh`.

### What You'll See

After all tasks complete, you'll see output like this:

```
============================================================
EVALUATION SUMMARY
============================================================
Total Tasks:       26
Successful:        5
Failed:            21
Success Rate:      19.23%
============================================================
```

This summary is:
- ✅ Displayed in the console
- ✅ Logged to the log file
- ✅ Saved to `score_summary.json` in the results directory

### Example Run

```bash
cd CoMEM-Agent-Inference

# Run evaluation (success rate displayed automatically at the end)
./run_agent.sh --eval_type webvoyager --domain Amazon --model qwen2.5-vl
```

---

## Manual Computation (Optional)

If you want to compute success rates manually for existing results, use the `compute_success_rate.py` script.

### Basic Usage

```bash
cd CoMEM-Agent-Inference

# Compute for most recent run
python compute_success_rate.py --recent 1

# Compute for specific evaluation type
python compute_success_rate.py --eval_type webvoyager --recent 1

# Compute for specific domain
python compute_success_rate.py --eval_type mmina --domain shopping --recent 1

# Compute for specific result directory
python compute_success_rate.py --result_dir results/webvoyager/Amazon/qwen2.5-vl/20251126_021327
```

### Save Results to JSON

```bash
# Compute and save to score_summary.json
python compute_success_rate.py --eval_type webvoyager --domain Amazon --save

# Show detailed per-task results
python compute_success_rate.py --eval_type webvoyager --domain Amazon --verbose
```

### Options

| Option | Description |
|--------|-------------|
| `--result_dir DIR` | Compute for specific result directory |
| `--eval_type TYPE` | Filter by evaluation type (mmina, webvoyager, mind2web) |
| `--domain DOMAIN` | Filter by domain (shopping, Amazon, etc.) |
| `--model MODEL` | Filter by model (qwen2.5-vl, etc.) |
| `--recent N` | Process N most recent results (default: 1) |
| `--save` | Save summary to `score_summary.json` |
| `--verbose, -v` | Print detailed per-task results |

### Examples

**Compare multiple recent runs:**
```bash
python compute_success_rate.py --eval_type webvoyager --domain Amazon --recent 3
```

**Compute for all shopping tasks:**
```bash
python compute_success_rate.py --eval_type mmina --domain shopping --save --verbose
```

**Check specific result directory:**
```bash
python compute_success_rate.py \
    --result_dir results/mmina/shopping/qwen2.5-vl/20251120_032543 \
    --verbose
```

---

## Output Format

### Console Output

```
============================================================
Evaluation Results: results/webvoyager/Amazon/qwen2.5-vl/20251126_021327
============================================================
Total Tasks:       26
Successful:        5
Failed:            21
Success Rate:      19.23%
============================================================
```

### JSON Output (`score_summary.json`)

```json
{
    "total_tasks": 26,
    "successful_tasks": 5,
    "failed_tasks": 21,
    "success_rate": 19.23,
    "scores": {
        "1": 1.0,
        "11": 0.0,
        "12": 0.0,
        "26": 1.0,
        ...
    }
}
```

---

## How It Works

### Score Extraction

The system parses HTML render files (`render_*.html`) to extract scores:

1. **Pattern matching**: Searches for `Score: X.X` or `Final Score: X.X` in HTML
2. **Pass/Fail detection**: If pattern not found, checks for `PASS` or `FAIL` keywords
3. **Task ID extraction**: Extracts task ID from filename (e.g., `render_26.html` → task ID `26`)

### Success Criteria

- A task is considered **successful** if score ≥ 0.5
- A task is considered **failed** if score < 0.5

### Automatic Integration

The test runners automatically call `_compute_and_display_success_rate()` after all tasks complete:

- **WebVoyager**: `benchmarks/webvoyager_evaluation/test_runner.py:355`
- **MMInA**: `benchmarks/MMInA_evaluation/test_runner.py:104`
- **Mind2Web**: (To be added)

---

## Troubleshooting

### No render files found

**Problem**: Script reports "No render files found"

**Solution**:
```bash
# Check if render files exist
ls results/webvoyager/Amazon/qwen2.5-vl/20251126_021327/render_*.html

# If files exist but not detected, check file names
ls -la results/webvoyager/Amazon/qwen2.5-vl/20251126_021327/
```

### Success rate is 0% but tasks passed

**Problem**: All tasks show as failed even though they passed

**Solution**: The HTML render files may not contain score information. Check a sample render file:

```bash
# View render file content
grep -i "score\|pass\|fail" results/.../render_1.html
```

If no score is found, the evaluation may have failed to write scores to HTML.

### Different success rates for same run

**Problem**: Manual computation shows different results than automatic

**Solution**: Ensure you're comparing the same set of tasks. Automatic computation runs after ALL tasks, while manual computation might run on partial results.

---

## Integration with Evaluation Pipeline

### Workflow

1. **Run evaluation**: `./run_agent.sh --eval_type mmina --domain shopping`
2. **Tasks execute**: Agent processes each task, scores are computed by evaluator
3. **Scores written**: Scores are written to HTML render files
4. **Automatic summary**: After all tasks complete, `_compute_and_display_success_rate()` runs
5. **Results saved**: Summary is saved to `score_summary.json` and displayed

### Files Created

For each evaluation run in `results/{eval_type}/{domain}/{model}/{datetime}/`:

```
20251126_021327/
├── config.json                    # Evaluation configuration
├── render_1.html                  # Task 1 results (with score)
├── render_11.html                 # Task 11 results
├── render_26.html                 # Task 26 results
├── score_summary.json             # SUCCESS RATE SUMMARY (NEW!)
└── logs/
    └── run.log                    # Detailed logs
```

---

## Related Files

- `compute_success_rate.py` - Standalone script for manual computation
- `benchmarks/webvoyager_evaluation/test_runner.py` - WebVoyager test runner
- `benchmarks/MMInA_evaluation/test_runner.py` - MMInA test runner
- `utils/help_functions.py` - Helper utilities (contains `save_scores_to_json`)

---

## Future Enhancements

Potential improvements to the success rate system:

1. **Real-time progress tracking**: Show success rate after each task
2. **Per-domain breakdown**: Compute success rates per domain for multi-domain runs
3. **Trend analysis**: Compare success rates across multiple runs
4. **Error categorization**: Break down failures by error type
5. **Visualization**: Generate charts and graphs of success rates

---

## Questions?

If success rates are not displaying correctly:

1. Check that you're using the latest code (test runners with `_compute_and_display_success_rate`)
2. Verify render files contain score information
3. Run manual computation with `--verbose` to see per-task details
4. Check logs for any errors during evaluation

For more information, see:
- `CLAUDE.md` - Project overview and usage
- `README.md` - Main project documentation
- `run_agent.sh --help` - Command-line options
