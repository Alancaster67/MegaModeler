import hydra
from hydra.utils import instantiate
from hydra import compose, initialize
import tensorflow as tf
from omegaconf import OmegaConf, open_dict
import pathlib
from flatten_dict import flatten, unflatten 
import optuna
from optuna.integration import TFKerasPruningCallback
from statistics import median
import functools
from optuna.integration.tensorboard import TensorBoardCallback

OmegaConf.register_new_resolver("eval", eval)

def read_conf():
    with initialize(version_base=None, config_path= "./Config"):
        cfg = compose(config_name="config")
        OmegaConf.resolve(cfg)
        cfg = cfg.finalconf
    return cfg

def hp_override(trial, hp_dict, hp):
    if hp_dict[hp]['type'] == 'float':
        hp_dict[hp]['value'] = trial.suggest_float(hp_dict[hp]['name'], hp_dict[hp]['domain'][0],hp_dict[hp]['domain'][1])
    
    elif hp_dict[hp]['type'] == 'int':
        hp_dict[hp]['value'] = trial.suggest_int(hp_dict[hp]['name'], hp_dict[hp]['domain'][0],hp_dict[hp]['domain'][1])
    
    elif hp_dict[hp]['type'] == 'cat':
        hp_dict[hp]['value'] = trial.suggest_categorical(hp_dict[hp]['name'], hp_dict[hp]['domain'])
    
    with open_dict(hp_dict):    
        hp_dict[hp] = hp_dict[hp]['value']
       
def hp_sample(trial, cfg):
    for k in cfg:
        if k[0] != '^' and OmegaConf.is_dict(cfg[k]):
            hp_sample(trial = trial, cfg = cfg[k])
            
        elif k[0] == '^' and OmegaConf.is_dict(cfg[k]):
            hp_override(trial = trial, hp_dict = cfg, hp = k)

def configure_architecture(trial, cfg): 
    
    #make a layer list from the cfg architecture
    blocks = [b for b in cfg.architecture]
    layers = {}
    for b in blocks:
        block_count = trial.suggest_int(b + 'BlockCount',cfg.architecture[b].mincount,cfg.architecture[b].maxcount)
        for i in range(block_count):
            for l in cfg.architecture[b].block:
                layer_key =F'{b}_{i+1}of{block_count}_L{len(layers) + 1}'
                layers[layer_key] = l
                
                if 'keras.applications.' in l['_target_'] and 'preprocess_input' not in l['_target_']:
                    
                    with open_dict(cfg):
                        cfg['transfer_learning'] = cfg.architecture[b]['pretrained']  
                        cfg['transfer_learning']['basemodel_key'] = layer_key
                    
    
    #append block and layernum to the begining of each hp name to avoid name clashing
    layers = flatten(layers)
    for k,v in layers.items():
        if k[-1] == 'name' and k[-2][0] =='^':
            layers[k] = k[0] + '_' + v
        
    cfg.architecture = unflatten(layers)

def configure_callbacks(trial,cfg):
    cbs = instantiate(cfg.callbacks)
    
    #configure special cases here
    for id,cb in enumerate(cbs):
        if isinstance(cb, functools.partial):
            if 'callbacks.ModelCheckpoint' in str(cb.func):
                checkpoint_path = pathlib.Path.cwd() / 'ModelCheckpoints' / cfg.optuna.study_name
                checkpoint_path = str(checkpoint_path) + f'/trial{trial.number}-' + 'Epoch{epoch:02d}-ValAcc{val_accuracy:.2f}.hdf5'
                checkpoint_cb = cb(filepath = checkpoint_path)
                cbs[id] = checkpoint_cb
    
    if cfg.optuna.pruning.enabled: #adding optuna pruning callback
        cbs.append(TFKerasPruningCallback(trial=trial, monitor= cfg.optuna.pruning.monitor))
    
    return cbs
  
def correct_config(cfg):
    cfg = flatten(OmegaConf.to_container(cfg))
    
    incorrect_keys = [k for k,_ in cfg.items() if k[-1][0] == '^']
    
    for ik in incorrect_keys:
        corrected_key = ik[:-1] + (ik[-1][1:],)
        cfg[corrected_key] = cfg.pop(ik)
    
    cfg = unflatten(cfg)    
    return OmegaConf.create(cfg)

def trialmodel(trial, cfg):
    
    tf.keras.backend.clear_session()
    
    if cfg.transfer_learning != None:
        transfer_learning = True
    else:
        transfer_learning = False
        
    train_dir = cfg.data.train_dataset
    val_dir = cfg.data.validation_dataset
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode = 'binary',
        image_size = (cfg.data.input_shape[0], cfg.data.input_shape[1]),
        batch_size = cfg.data.batch_size
        )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode = 'binary',
        image_size = (cfg.data.input_shape[0], cfg.data.input_shape[1]),
        batch_size = cfg.data.batch_size
        )
    
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(train_ds.cardinality()).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    #if transfer_learning:
    #    def preprocess(images, labels):
    #        prep_fun from optuna.integration.tensorboard import TensorBoardCallback= instantiate(cfg.transfer_learning.preprocess_function, _partial_=True)
    #        return prep_fun(images), labels
        
    #    train_ds = train_ds.map(preprocess)
    #    val_ds = val_ds.map(preprocess)
    
    #model = tf.keras.Sequential()
    #for k in cfg.architecture:
    #    layer = instantiate(cfg.architecture[k])
        
    #    if transfer_learning:
    #        if k == cfg.transfer_learning.basemodel_key:
    #            layer.trainable = False
    #            base_model = layer

    #    model.add(layer)
    
    #Using the functional API for flexibility          
    model_inputs = instantiate(cfg.input)
    for count, k in enumerate(cfg.architecture, start=1):
        layer = instantiate(cfg.architecture[k])
        
        if count == 1: #input to the DAG
                layer_inputs = model_inputs
        
        if transfer_learning == False:
            if count == len(cfg.architecture): #output of the DAG
                output = layer(layer_inputs)
            else:
                layer_inputs = layer(layer_inputs) #hidden layers of the DAG
        else:
            if k == cfg.transfer_learning.basemodel_key:
                layer.trainable = False
                base_model = layer
                
                if count == len(cfg.architecture): #output of the DAG where the pretrained model is the last block
                    output = layer(layer_inputs, training = False)
                else:
                    layer_inputs = layer(layer_inputs, training = False)
            else:
                if count == len(cfg.architecture):
                    output = layer(layer_inputs)
                else:
                    layer_inputs = layer(layer_inputs)
    
    model = tf.keras.Model(model_inputs, output)
    model.compile(optimizer = instantiate(cfg.optimizer),
                loss = instantiate(cfg.loss),
                metrics = cfg.metric.primary)   
    
    cbs = configure_callbacks(trial = trial, cfg = cfg)
    
    #cbs = instantiate(cfg.callbacks)
    #if cfg.optuna.pruning.enabled:
    #    cbs.append(TFKerasPruningCallback(trial=trial, monitor= cfg.optuna.pruning.monitor))
    
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = cfg.training.epochs,
        callbacks = cbs
        )
    
    if transfer_learning and cfg.transfer_learning.fine_tuning.unfreeze_top_xlayers != None:
        if cfg.transfer_learning.fine_tuning.unfreeze_top_xlayers >0:
            for layer in base_model.layers[(-1*cfg.transfer_learning.fine_tuning.unfreeze_top_xlayers):]:
                layer.trainable = True
        
        optimizer_key = [k for k in cfg.optimizer][0]
        
        lr = (cfg.optimizer.learning_rate)/(cfg.transfer_learning.fine_tuning.lr_divisor)
        model.compile(optimizer = instantiate(cfg.optimizer, learning_rate = lr),
            loss = instantiate(cfg.loss),
            metrics = cfg.metric.primary)
        
        total_epochs =  cfg.training.epochs + cfg.transfer_learning.fine_tuning.additional_epochs

        history_fine = model.fit(train_ds,
                                epochs=total_epochs,
                                initial_epoch=history.epoch[-1],
                                validation_data=val_ds)
        
        
        history.history['accuracy'] += history_fine.history['accuracy']
        history.history['val_accuracy'] += history_fine.history['val_accuracy']
        history.history['loss'] += history_fine.history['loss']
        history.history['val_loss'] += history_fine.history['val_loss']


    # TODO Need to couple this to the config value somehow
    return median(history.history['val_accuracy'][-3:])

def main(trial):
    cfg = read_conf()
    configure_architecture(trial = trial, cfg = cfg)
    hp_sample(trial = trial, cfg = cfg)
    new_cfg = correct_config(cfg = cfg)
    opti_metric = trialmodel(trial = trial, cfg = new_cfg)
    
    return opti_metric

if __name__ == "__main__":
    with initialize(version_base=None, config_path= "./Config/optunaconf"):
        opt_cfg = compose(config_name="optuna")
    
    pruner = None
    if opt_cfg.pruning.enabled:
        pruner = instantiate(opt_cfg.pruning.pruner)
    
    tensorboard_callback = TensorBoardCallback(f"TensorBoardLogs/{opt_cfg.study_name}/", metric_name="accuracy")
    
    study = optuna.create_study(
        study_name = opt_cfg.study_name,
        storage = opt_cfg.storage,
        load_if_exists = True,
        direction = opt_cfg.direction,
        pruner = pruner,
        )
    study.optimize(
        main,
        n_trials = opt_cfg.trials,
        catch = (
            tf.errors.ResourceExhaustedError,
            tf.errors.NotFoundError
            ),
        gc_after_trial = True,
        callbacks = [tensorboard_callback]
        )
