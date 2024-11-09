# Training language model for classification
### Entering tasks/classification

Model training: 
```python
python train.py --config ./configs/base.json
```
The result would be saved in a file specified by the config 
![image](https://github.com/user-attachments/assets/8e1426f2-8678-4d70-a9b9-cf9e5b23f2b6)

Inference:
```python
python test_inference.py --config ./configs/base.json
```

Making config grid-search:
```python
python make_configs.py
```

### If you want to change the config file, create a new one in the configs folder.
What a config file might look like

![image](https://github.com/user-attachments/assets/c654c2ab-3db6-4c38-bcd3-c902153488e1)
