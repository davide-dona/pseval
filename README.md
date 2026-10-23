# pseval

Evaluation protocol for probabilistic suffix prediction in predictive process monitoring.

```bash
uv add git+https://github.com/davide-dona/pseval@v1.0.0
```

```python
import pseval

benchmark = pseval.load_benchmark('sepsis')
scores = pseval.evaluate(
    model, benchmark, directory, selection=pseval.Selection.none(), label='mine'
)
print(pseval.read_scores(scores).summary().overall)
print(pseval.skill(pseval.read_scores(scores), benchmark))
```
