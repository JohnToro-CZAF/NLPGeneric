import os 
import json 
from itertools import product 
 
# base_config = { 
#     "model_config": { 
#         "model_type": "BiDeepRNN", 
#         "args": { 
#             "dim_input": 200, 
#             "dim_hidden": 128, 
#             "num_layers": 1, 
#             "dim_output": 2, 
#             "embedding_strategy": "word2vec", 
#             "pretrained_path": "glove-twitter-200", 
#             "embedding_frozen": True, 
#             "oov_handling": "using_unk",
#             # "aggregation": "last" 
#         } 
#     }, 
#     "tokenizer_config": { 
#         "tokenizer_type": "nltk", 
#         "args": { 
#             "dataset": "rotten_tomatoes" 
#         } 
#     }, 
#     "trainer_args": { 
#         "task": "classification", 
#         "training_batch_size": 32, 
#         "validation_batch_size": 32, 
#         "learning_rate": 0.0001, 
#         "epoch": 100 
#     }, 
#     "metric_config": { 
#         "metrics": [ 
#             {"name": "accuracy", "args": {}}, 
#             {"name": "f1", "args": {}}, 
#             {"name": "precision", "args": {}}, 
#             {"name": "recall", "args": {}} 
#         ] 
#     }, 
#     "data_config": { 
#         "name": "rotten_tomatoes", 
#         "is_huggingface": True, 
#         "type": "classification" 
#     }, 
#     "analysis_config": { 
#         "output_dir": "experiments/n_layers/bigru_layer=1", 
#         "record_metrics": True, 
#         "record_gradients": True, 
#         "save_interval": 1000 
#     } 
# } 
 

# RNN AGGREGATION RUN 
# Parameters to vary 
# batch_sizes = [128] 
# learning_rates = [0.00001] 
# architectures = ['DeepRNN', 'BiDeepRNN'] 
# num_layers_list = [1, 3, 5] 
# dim_hidden = [128]
# aggregations = ['last', 'mean', 'max', 'attention']
 
# config_dir = './configs/conf_folder/' 
 
# os.makedirs(config_dir, exist_ok=True) 
 
# shell_script_lines = [] 
 
# for batch_size, learning_rate, architecture, num_layers, aggregation, dh in product(batch_sizes, learning_rates, architectures, num_layers_list, aggregations, dim_hidden): 
#     config = json.loads(json.dumps(base_config)) 
     
#     config['model_config']['model_type'] = architecture 
#     config['model_config']['args']['num_layers'] = num_layers 
#     config['model_config']['args']['dim_hidden'] = dh

#     config['model_config']['args']['aggregation'] = aggregation
#     config['trainer_args']['training_batch_size'] = batch_size 
#     config['trainer_args']['validation_batch_size'] = batch_size 
#     config['trainer_args']['learning_rate'] = learning_rate 
     
#     output_dir = f"experiments/{architecture}_layers={num_layers}_bs={batch_size}_lr={learning_rate}_aggregation={aggregation}" 
#     config['analysis_config']['output_dir'] = output_dir 
     
#     config_filename = f"{architecture}_layers={num_layers}_bs={batch_size}_lr={learning_rate}_aggregation={aggregation}.json" 
#     config_filepath = os.path.join(config_dir, config_filename) 
     
#     with open(config_filepath, 'w') as f: 
#         json.dump(config, f, indent=4) 
     
#     shell_script_lines.append(f"python train.py --config {config_filepath}") 
     


base_config = { 
    "model_config": { 
        "model_type": "BiDeepRNN", 
        "args": { 
            "dim_input": 200, 
            "dim_hidden": 128, 
            "n_layers": 1, 
            "n_label": 2, 
            "embedding_strategy": "word2vec", 
            "pretrained_path": "glove-twitter-200", 
            "embedding_frozen": False, 
            "oov_handling": "using_unk",
            # "aggregation": "last" 
        } 
    }, 
    "tokenizer_config": { 
        "tokenizer_type": "nltk", 
        "args": { 
            "dataset": "rotten_tomatoes" 
        } 
    }, 
    "trainer_args": { 
        "task": "classification", 
        "training_batch_size": 32, 
        "validation_batch_size": 32, 
        "learning_rate": 0.0001, 
        "epoch": 100 
    }, 
    "metric_config": { 
        "metrics": [ 
            {"name": "accuracy", "args": {}}, 
            {"name": "f1", "args": {}}, 
            {"name": "precision", "args": {}}, 
            {"name": "recall", "args": {}} 
        ] 
    }, 
    "data_config": { 
        "name": "rotten_tomatoes", 
        "is_huggingface": True, 
        "type": "classification" 
    }, 
    "analysis_config": { 
        "output_dir": "experiments/n_layers/bigru_layer=1", 
        "record_metrics": True, 
        "record_gradients": True, 
        "save_interval": 1000 
    } 
} 
## CAPSULE EXPERIMENT
batch_sizes = [128] 
learning_rates = [0.00001] 
architectures = ['EncoderRNN'] 
num_layers_list = [1, 3, 5] 
bidrectional = [True, False]
dim_hidden = [128]
dropout_rate = [0.15]
rnn_type = ['RNN', 'LSTM', 'GRU']
oov_handling = ['using_unk', 'average_context']

config_dir = './configs/capsule_conf_folder_train_embeddings/' 
 
os.makedirs(config_dir, exist_ok=True) 
 
shell_script_lines = [] 
 
for batch_size, learning_rate, architecture, num_layers, bi, dh, dropout, rnn, oov in product(batch_sizes, learning_rates, architectures, num_layers_list, bidrectional, dim_hidden, dropout_rate, rnn_type, oov_handling): 
    config = json.loads(json.dumps(base_config)) 
     
    config['model_config']['model_type'] = architecture 
    config['model_config']['args']['n_layers'] = num_layers 
    config['model_config']['args']['dim_hidden'] = dh

    config['model_config']['args']['bidirectional'] = bi
    config['model_config']['args']['cell_dropout_rate'] = dropout
    config['model_config']['args']['embed_dropout_rate'] = dropout
    config['model_config']['args']['final_dropout_rate'] = dropout

    config['model_config']['args']['rnn_type'] = rnn
    config['model_config']['args']['oov_handling'] = oov
    if oov=='average_context':
        config['model_config']['args']['context_window'] = 5







    config['trainer_args']['training_batch_size'] = batch_size 
    config['trainer_args']['validation_batch_size'] = batch_size 
    config['trainer_args']['learning_rate'] = learning_rate 
     
    output_dir = f"experiments_capsule_no_dropout/{architecture}_layers={num_layers}_bidirectional={str(bi)}_lr={learning_rate}_dropout={dropout}_rnn={rnn}_oov={oov}" 
    config['analysis_config']['output_dir'] = output_dir 
     
    config_filename = f"{architecture}_layers={num_layers}_bidirectional={str(bi)}_lr={learning_rate}_dropout={dropout}_rnn={rnn}_oov={oov}.json" 
    config_filepath = os.path.join(config_dir, config_filename) 
     
    with open(config_filepath, 'w') as f: 
        json.dump(config, f, indent=4) 
     
    shell_script_lines.append(f"python train.py --config {config_filepath}")     
shell_script_content = '\n'.join(shell_script_lines) 
shell_script_path = 'run_all_configs_capsulernn_train_embeddings.sh' 
 
with open(shell_script_path, 'w') as f: 
    f.write(shell_script_content) 
 
os.chmod(shell_script_path, 0o755) 
 
print(f"Generated {len(shell_script_lines)} configurations.") 
print(f"Configurations are saved in {config_dir}") 
print(f"Shell script to run all configurations is saved as {shell_script_path}")