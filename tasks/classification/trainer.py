import os
import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
import metrics
from metrics import beautify

SUPPORTED_TASKS = ["classification", "causal"]

class BaseLossFunction(nn.Module):
  def __init__(self):
    super(BaseLossFunction, self).__init__()

  def forward(self, input, output, label):
    raise NotImplementedError("forward method should be implemented")

class ClassificationLossFunction(BaseLossFunction):
  def __init__(self):
    super(ClassificationLossFunction, self).__init__()

  def forward(self, output, label):
    # input : (batch_size, seq_len), output : (batch_size, seq_len, num_classes), label : (batch_size)
    # get the (batch_size) tensor of positions that is different from padding token
    return F.cross_entropy(output, label)

def get_loss_fn(task: str):
  if task == "classification":
    return ClassificationLossFunction()
  else:
    raise NotImplementedError(f"Task {task} not implemented")

class TrainingArgs:
  def __init__(
      self, 
      task: str,
      learning_rate: float, 
      training_steps: int,
      metric_log_interval: int,
      eval_interval: int,
      training_batch_size: int,
      validation_batch_size: int,
      early_stopper,
    ):
    """ Training Arguments for the Trainer class

    Args:
        task (str): name of the task
        learning_rate (float): learning rate for the optimizer
        training_steps (int): number of training steps
        metric_log_interval (int): how many steps to wait before logging metrics
        training_batch_size (int): training batch size
        validation_batch_size (int): validation batch size
    """
    assert task in SUPPORTED_TASKS, f"task should be one of {SUPPORTED_TASKS}"
    assert metric_log_interval <= training_steps, "metric_log_interval should be less than or equal to training"
    self.task = task
    self.learning_rate = learning_rate
    self.training_steps = training_steps
    self.eval_interval = eval_interval
    self.metric_log_interval = metric_log_interval
    self.training_batch_size = training_batch_size
    self.validation_batch_size = validation_batch_size
    self.early_stopper = early_stopper

class Trainer:
  def __init__(
      self, 
      model: nn.Module, 
      training_args: TrainingArgs, 
      train_loader: torch.utils.data.DataLoader,
      val_loader: torch.utils.data.DataLoader,
      optimizer: torch.optim.Optimizer,
      metric_names: list[str],
      analysis_config: dict
    ):
    self.args = training_args
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.optimizer = optimizer
    self.loss_fn = get_loss_fn(self.args.task)
    self.metric_names = metric_names
    self.analysis_config = analysis_config
    self.metrics_log = {
        'train_loss': [],
        'train_metrics': {metric['name']: [] for metric in self.metric_names},
        'val_loss': [],
        'val_metrics': {metric['name']: [] for metric in self.metric_names},
        'steps': [],
        'val_steps': []  # Add this line
    }
    self.output_dir = self.analysis_config.get('output_dir', 'output/exp1')
    os.makedirs(self.output_dir, exist_ok=True)
  
  def get_metrics_dict(self):
    return {metric["name"]: metrics.build(metric["name"], metric["args"]) for metric in self.metric_names}

  def eval_step(self, input, length, label):
    input = input.to("cuda")
    with torch.no_grad():
      output = self.model(input)
    output = output.to("cpu")
    # outputs : (batch_size, seq_len, num_classes)
    # result : (batch_size, num_classes)
    output = output[range(input.size()[0]), length - 1]
    loss = self.loss_fn(output, label)
    return output, loss.item()

  def eval(self):
    val_loss = []
    eval_metrics_dict = self.get_metrics_dict()
    for input, length, label in self.val_loader:
      output, loss = self.eval_step(input, length, label)
      val_loss.append(loss/input.size()[0])
      for metric_name, metric in eval_metrics_dict.items():
        metric.update(output, label)
    
    avg_val_loss = sum(val_loss) / len(val_loss)
    result_metrics = {
      metric_name: metric.value() for metric_name, metric in eval_metrics_dict.items()
    }
    
    # ! For printing
    print(
      f"""Validating result:
        Validation Loss: {avg_val_loss},
        Metrics: {beautify(result_metrics)}"""
    )
    
    # ! For logging analysis
    self.metrics_log['val_loss'].append(avg_val_loss)
    for metric_name, value in result_metrics.items():
        self.metrics_log['val_metrics'][metric_name].append(value)
      
    if self.early_stopper.early_stop(result_metrics['accuracy']):
      print('Early Stopping activated')
      return True

    return False
  
  def train_step(self, input, length, label):
    self.optimizer.zero_grad()
    input = input.to("cuda")
    output = self.model(input) # output : (batch_size, seq_len, num_classes)
    output = output.to("cpu")
    output = output[range(input.size()[0]), length - 1]
    loss = self.loss_fn(output, label)
    loss.backward()
    self.optimizer.step()
    
    # Record gradients if required
    if self.analysis_config.get('record_gradients', False):
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item()
        self.metrics_log.setdefault('grad_norms', []).append(grad_norm)
    return output, loss.item()

  def train(self):
    self.model.train()
    train_loss = 0
    data_metrics_dict = self.get_metrics_dict()
    print("Data Metrics: ", data_metrics_dict)
    data_iter = iter(self.train_loader)
    for step_id in tqdm.tqdm(range(self.args.training_steps)):
      epoch_loss = []
      epoch_metrics_dict = self.get_metrics_dict()
      for i, (input, length, label) in tqdm(enumerate(train_loader)):
        output, loss = self.train_step(input, length, label) # output : (batch_size, seq_len, num_classes)
        epoch_loss.append(loss / input.size()[0])
        for metric_name, metric in epoch_metrics_dict.items():
          metric.update(output, label) 
      
      avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
      train_loss += avg_epoch_loss
      result_metrics = {
        metric_name: metric.value() for metric_name, metric in epoch_metrics_dict.items()
      }
      # ! For logging analysis
      for metric_name, metric in epoch_metrics_dict.items():
          value = metric.value()
          self.metrics_log['train_metrics'][metric_name].append(value)
      self.metrics_log['epoch'].append(step_id + 1)
      self.metrics_log['train_loss'].append(avg_epoch_loss)
      
      # ! For printing
      result_metrics = {
        metric_name: metric.value() for metric_name, metric in data_metrics_dict.items()
      }
      print(
        f"""Epoch {step_id + 1}:
            Train Loss: {train_loss / (step_id + 1)},
            Metrics:{beautify(result_metrics)}"""
      )
      es = self.eval()
      self.metrics_log['val_steps'].append(step_id + 1)  # Record validation step
    
      self.save_metrics()

      if es:
        break
        
  def save_metrics(self):
      # Save metrics to a JSON file
      metrics_file = os.path.join(self.output_dir, 'metrics.json')
      with open(metrics_file, 'w') as f:
          json.dump(self.metrics_log, f, indent=4)

      # Generate and save plots
      steps = self.metrics_log['steps']
      val_steps = self.metrics_log['val_steps']
      # Plot training loss
      plt.figure()
      plt.plot(steps, self.metrics_log['train_loss'], label='Training Loss')
      plt.xlabel('Steps')
      plt.ylabel('Loss')
      plt.title('Training Loss over Time')
      plt.legend()
      plt.savefig(os.path.join(self.output_dir, 'training_loss.png'))
      plt.close()
      
      plt.figure()
      plt.plot(val_steps, self.metrics_log['val_loss'], label='Validation Loss')
      plt.xlabel('Steps')
      plt.ylabel('Loss')
      plt.title('Validation Loss over Time')
      plt.legend()
      plt.savefig(os.path.join(self.output_dir, 'validation_loss.png'))
      plt.close()
      
      # Plot training and validation loss
      plt.figure()
      plt.plot(steps, self.metrics_log['train_loss'], label='Training Loss')
      plt.plot(val_steps, self.metrics_log['val_loss'], label='Validation Loss')
      plt.xlabel('Steps')
      plt.ylabel('Loss')
      plt.title('Training and Validation Loss over Time')
      plt.legend()
      plt.savefig(os.path.join(self.output_dir, 'loss.png'))
      plt.close()
      
      # Plot metrics
      for metric_name in self.metrics_log['train_metrics']:
          plt.figure()
          plt.plot(steps, self.metrics_log['train_metrics'][metric_name], label=f'Train {metric_name}')
          plt.plot(val_steps, self.metrics_log['val_metrics'][metric_name], label=f'Validation {metric_name}')
          plt.xlabel('Steps')
          plt.ylabel(metric_name)
          plt.title(f'{metric_name.capitalize()} over Time')
          plt.legend()
          plt.savefig(os.path.join(self.output_dir, f'{metric_name}.png'))
          plt.close()

      # Save gradient norms if recorded
      if 'grad_norms' in self.metrics_log:
          plt.figure()
          plt.plot(steps, self.metrics_log['grad_norms'], label='Gradient Norm')
          plt.xlabel('Steps')
          plt.ylabel('Gradient Norm')
          plt.title('Gradient Norm over Time')
          plt.legend()
          plt.savefig(os.path.join(self.output_dir, 'gradient_norm.png'))
          plt.close()
