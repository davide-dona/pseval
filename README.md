# pseval

Evaluation protocol for probabilistic suffix prediction in predictive process monitoring.

```bash
uv add git+https://github.com/davide-dona/pseval@v1.0.0
```

Train your model on `benchmark.train` and `benchmark.validation` however you like, comparing
candidates with `pseval.validate`. pseval only calls its `sample`, then scores the test draws.

```python
import pseval

benchmark = pseval.load_benchmark('sepsis')
model = train(benchmark.train, benchmark.validation)  # yours
print(pseval.validate(model, benchmark)['signature_kernel_score'])

generations = pseval.generate(
    model,
    benchmark,
    directory / 'generations.parquet',
    label='mine',
)
with pseval.ScoresReader(pseval.score(generations, benchmark)) as reader:
    scores = reader.read()
print(scores.summary().overall)
print(pseval.skill(scores, benchmark))
```
