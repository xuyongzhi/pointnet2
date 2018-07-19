import tensorflow as tf
import os,sys

BASE_DIR = os.path.dirname( os.path.abspath(__file__) )
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'models'))
LOG_DIR = os.path.join(BASE_DIR, 'log')

from pointnet2_cls_ssg import get_loss, get_model_single_scale
from modelnet_feed import input_fn_h5_, input_fn_np

NUM_EPOCHS = 101
gpu_id = 1
input_function = input_fn_h5_
DATA_DIR = None
BATCH_SIZE = 32


def model_fn(features, labels, mode, params):
  bn_decay = 0.5
  learning_rate = 0.001

  is_training = mode == tf.estimator.ModeKeys.TRAIN
  is_training = tf.cast(is_training, tf.bool)
  logits, end_points = get_model_single_scale(features, is_training, bn_decay)
  labels = tf.squeeze(labels,1)
  loss = get_loss(logits, labels, end_points)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  global_step = tf.train.get_or_create_global_step()
  minimize_op = optimizer.minimize(loss, global_step)
  train_op = tf.group(minimize_op)

  classes = tf.argmax(logits, axis=-1)
  predictions = {'classes': classes}
  accuracy = tf.metrics.accuracy(labels, classes)
  metrics = {'accuracy': accuracy}

  return tf.estimator.EstimatorSpec(
    mode = mode,
    predictions = predictions,
    loss = loss,
    train_op = train_op,
    eval_metric_ops = metrics )


def input_fn_train():
  return input_function(
      is_training=True, data_dir=DATA_DIR,
      batch_size=BATCH_SIZE,
      data_net_configs = None,
      num_epochs=1)

def input_fn_eval():
  return input_function(
      is_training=False, data_dir=DATA_DIR,
      batch_size=BATCH_SIZE,
      data_net_configs = None,
      num_epochs=1)

def train_main():
  session_config = tf.ConfigProto(
      inter_op_parallelism_threads=0,
      intra_op_parallelism_threads=0,
      gpu_options = tf.GPUOptions(allow_growth = True),
      allow_soft_placement=True)
  distribution = tf.contrib.distribute.OneDeviceStrategy('device:GPU:%d'%(gpu_id))
  run_config = tf.estimator.RunConfig(train_distribute=distribution,
                                      session_config=session_config)
  classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=LOG_DIR, config=run_config,
    params={})

  log_fn = os.path.join(LOG_DIR, 'log_metric.txt')
  with open(log_fn,'w') as logf:
    for cycle in range(NUM_EPOCHS):
      classifier.train(input_fn=input_fn_train)
      train_eval_results = classifier.evaluate(input_fn=input_fn_train,
                        steps=None, name='train')
      eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                          steps=None,
                                          name='test')
      acu_str = '\n{} train {:.3f} eval {:.3f}\n'.format( cycle,
        train_eval_results['accuracy'], eval_results['accuracy'])
      print(acu_str)
      logf.write(acu_str)
      logf.flush()

if __name__ == '__main__':
  train_main()


