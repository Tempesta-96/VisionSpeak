# Epoch sweep on raw MP4 clips

- Speaker held out: speaker_test_samples
- Clips: 160
- Frames per clip: max 90, min 30, min detection rate 0.75
- Checkpoints: checkpoints/retrain
- Cache key: {sample_id}__min30_rate0.75.npz

## Overall test accuracy

| epoch | CTC + BiGRU | CTC + Transformer | Word + BiGRU | Word + Transformer |
| --- | --- | --- | --- | --- |
| 50 | 16.25% | 23.75% | 12.50% | 40.62% |
| 100 | 31.87% | 24.38% | 12.50% | 35.00% |
| 200 | 49.38% | 40.62% | 12.50% | 49.38% |

## Clips correct

| epoch | CTC + BiGRU | CTC + Transformer | Word + BiGRU | Word + Transformer |
| --- | --- | --- | --- | --- |
| 50 | 26/160 | 38/160 | 20/160 | 65/160 |
| 100 | 51/160 | 39/160 | 20/160 | 56/160 |
| 200 | 79/160 | 65/160 | 20/160 | 79/160 |
