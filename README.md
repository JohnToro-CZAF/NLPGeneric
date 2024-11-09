# Training language model for classification
### Entering tasks/classification

Model training: 
```python
python train.py --config ./configs/base.json
```

Inference:
```python
python test_inference.py --config ./configs/base.json
```

Making config grid-search:
```python
python make_configs.py
```

### If you want to change the config file, create a new one in the configs folder.
