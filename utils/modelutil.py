import shutil
import re
import layers
import tensorflow_core as tf
import math
from os import path, makedirs
import logger
from utils.kerasutil import ModelCallback
from utils.confutil import object_from_conf, register_conf
import keras.backend as K

# A fake call to register
register_conf(name="adam", scope="optimizer", conf_func=lambda conf: tf.keras.optimizers.Adam(**conf))(None)
register_conf(name="sgd", scope="optimizer", conf_func=lambda conf: tf.keras.optimizers.SGD(**conf))(None)

register_conf(name="exponential_decay", scope="learning_rate",
              conf_func=lambda conf: tf.keras.optimizers.schedules.ExponentialDecay(**conf))(None)

_MODE_RESUME = "resume"
_MODE_NEW = "new"
_MODE_RESUME_COPY = "resume-copy"


def layer_from_config(layer_conf, model_conf, data_conf):
    """
    Get the corresponding keras layer from configurations
    :param layer_conf: The layer configuration
    :param model_conf: The global model configuration, sometimes it is used to generate some
    special layer like "output-classification" and "output-segmentation" layer
    :param data_conf: The dataset configuration, for generating special layers
    :return: A keras layer
    """
    context = {"class_count": data_conf["class_count"]}
    return object_from_conf(layer_conf, scope="layer", context=context)

def optimizer_from_config(learning_rate, optimizer_conf):
    """
    Get the optimizer from configuration
    :param learning_rate: The learning rate, might be a scalar or a learning rate schedule
    :param optimizer_conf: The optimizer configuration
    :return: An corresponding optimizer
    """
    context = {"learning_rate": learning_rate}
    return object_from_conf(optimizer_conf, scope="optimizer", context=context)

def learning_rate_from_config(learning_rate_conf):
    """
    Get the learning rate scheduler based on configuration
    :param learning_rate_conf: The learning rate configuration
    :return: A learning rate scheduler
    """
    return object_from_conf(learning_rate_conf, scope="learning_rate")

def net_from_config(model_conf, data_conf):
    """
    Generate a keras network from configuration dict
    :param model_conf: The global model configuration dictionary
    :param data_conf: The configuration of the dataset, it might use to initialize some layer like
    "output-classification"
    :param train_dataset: The train dataset, used to add input layer based on shape
    :return: A keras net
    """
    # Get network conf
    net_conf = model_conf["net"]

    # Input layer
    transform_confs = model_conf["dataset"].get("train_transforms", [])
    # Get the shape of the dataset, first check whether we have clip-feature layer in the dataset, if not, we
    # use the feature size in the dataset configuration
    feature_size = None
    for transform_conf in transform_confs[::-1]:
        if type(transform_conf) is dict and transform_conf.get("name") == "clip-feature":
            feature_size = transform_conf["c"]
            logger.log("Get feature_size={} from model configuration".format(feature_size))
    if feature_size is None:
        feature_size = data_conf.get("feature_size")
        logger.log("Get feature_size={} from dataset configuration".format(feature_size))
    assert feature_size is not None, "Cannot determine the feature_size"
    # Get the point size, if possible
    point_count = data_conf.get("point_count")
    for transform_conf in transform_confs[::-1]:
        if type(transform_conf) is dict and transform_conf.get("name") == "sampling":
            point_count = None
            logger.log("Ignore point_count since we have transform sampling from dataset")
    # input_layer = tf.keras.layers.InputLayer(input_shape=(point_count, feature_size))

    # Extend feature layer
    if "extend_feature" in net_conf:
        logger.log("\"extend_feature\" is deprecated, use \"input-feature-extend\" layer instead", color="yellow")

    inputs = tf.keras.Input(shape=(point_count, feature_size))
    if net_conf["structure"] == "sequence":

        xyz_points_list = [[inputs[..., :3], inputs[..., 3:]]]
        
        # process SA layers
        for idx in range(4):
            layer_conf = net_conf["layers"][idx]
            logger.log(f"In constructing: {layer_conf}")
            layer = layer_from_config(layer_conf, model_conf, data_conf)
            output = layer(xyz_points_list[-1][0], xyz_points_list[-1][1])
            xyz_points_list.append([output[0], output[1]])

        sem_list = [xyz_points_list[-1][1]]

        # process FP layers
        for idx in range(4, 8):
            layer_conf = net_conf["layers"][idx]
            logger.log(f"In constructing: {layer_conf}")
            layer = layer_from_config(layer_conf, model_conf, data_conf)
            output = layer(xyz_points_list[7-idx][0], xyz_points_list[8-idx][0], xyz_points_list[7-idx][1], sem_list[-1])
            sem_list.append(output)

        layer_conf = net_conf["layers"][8]
        logger.log(f"In constructing: {layer_conf}")
        layer = layer_from_config(layer_conf, model_conf, data_conf)
        net_sem = layer(sem_list[-1])

        layer_conf = net_conf["layers"][9]
        logger.log(f"In constructing: {layer_conf}")
        layer = layer_from_config(layer_conf, model_conf, data_conf)
        net_sem_cache = layer(sem_list[-1])

        ins_list = [xyz_points_list[-1][1]]

        # process FP layers
        for idx in range(10, 14):
            layer_conf = net_conf["layers"][idx]
            logger.log(f"In constructing: {layer_conf}")
            layer = layer_from_config(layer_conf, model_conf, data_conf)
            output = layer(xyz_points_list[7-idx][0], xyz_points_list[8-idx][0], xyz_points_list[7-idx][1], ins_list[-1])
            ins_list.append(output)

        layer_conf = net_conf["layers"][14]
        logger.log(f"In constructing: {layer_conf}")
        layer = layer_from_config(layer_conf, model_conf, data_conf)
        net_ins = layer(ins_list[-1])  

        net_ins = net_ins + net_sem_cache       

        for idx in range(15, 17):
            layer_conf = net_conf["layers"][idx]
            logger.log(f"In constructing: {layer_conf}")
            layer = layer_from_config(layer_conf, model_conf, data_conf)
            net_ins = layer(net_ins)

        layer_conf = net_conf["layers"][17]
        logger.log(f"In constructing: {layer_conf}")
        layer = layer_from_config(layer_conf, model_conf, data_conf)
        adj_matrix = layer(net_ins)   

        layer_conf = net_conf["layers"][18]
        logger.log(f"In constructing: {layer_conf}")
        layer = layer_from_config(layer_conf, model_conf, data_conf)
        nn_idx = layer(adj_matrix)         

        layer_conf = net_conf["layers"][19]
        logger.log(f"In constructing: {layer_conf}")
        layer = layer_from_config(layer_conf, model_conf, data_conf)
        net_sem = layer(net_sem, nn_idx) 

        for idx in range(20, 22):
            layer_conf = net_conf["layers"][idx]
            logger.log(f"In constructing: {layer_conf}")
            layer = layer_from_config(layer_conf, model_conf, data_conf)
            net_sem = layer(net_sem)   
        
        # concatenate two output tensors
        # semantics label first
        outputs = tf.concat([net_sem, net_ins], -1)

        return tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        assert False, "\"{}\" is currently not supported".format(net_conf["structure"])

def my_loss_function(y_true, y_pred):
    # input shape: N * (num_of_classes + E)

    # split the predict value & actual label
    pred_sem = y_pred[:, :13]
    pred_ins = y_pred[:, 13:]
    true_sem = y_true[:, :13]
    true_ins = y_true[:, 13:]

    # classify loss

    # our predict result is log probability (without softmax process)
    # so we set from_logits = true
    # check https://keras.io/zh/backend/ for further detail
    classify_loss = K.sparse_categorical_crossentropy(true_sem, pred_sem, from_logits=True)

    # discriminative loss
    feature_dim = pred_ins.get_shape()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.
    param_reg = 0.001
    
    # 返回discriminative loss以及附带的三个参数
    disc_loss, l_var, l_dist, l_reg = discriminative_loss(pred_ins, true_ins, feature_dim,
                                         delta_v, delta_d, param_var, param_dist, param_reg)

    # total loss
    loss = classify_loss + disc_loss
    return loss

def discriminative_loss(prediction, correct_label, feature_dim,
                        delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Iterate over a batch of prediction/label and cumulate loss
    :return: discriminative loss and its three components
    '''

    # i: 第i个batch, i >= B时循环停止
    def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
        disc_loss, l_var, l_dist, l_reg = discriminative_loss_single(prediction[i], correct_label[i], feature_dim,
                                                                     delta_v, delta_d, param_var, param_dist, param_reg)
        # 在第i个index下写进后面的value
        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)
        out_reg = out_reg.write(i, l_reg)

        return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
    output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                   size=0,
                                   dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [correct_label,
                                                                                           prediction,
                                                                                           output_ta_loss,
                                                                                           output_ta_var,
                                                                                           output_ta_dist,
                                                                                           output_ta_reg,
                                                                                           0])
    # 将array的元素堆叠成tensor
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()
    out_reg_op = out_reg_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)
    l_reg = tf.reduce_mean(out_reg_op)

    return disc_loss, l_var, l_dist, l_reg

def discriminative_loss_single(prediction, correct_label, feature_dim,
                               delta_v, delta_d, param_var, param_dist, param_reg):
    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''

    ### Reshape so pixels are aligned along a vector
    #correct_label = tf.reshape(correct_label, [label_shape[1] * label_shape[0]])
    reshaped_pred = tf.reshape(prediction, [-1, feature_dim])

    ### Count instances
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)

    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)
    

    segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    mu_expand = tf.gather(mu, unique_id)

    ### Calculate l_var
    #distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    #tmp_distance = tf.subtract(reshaped_pred, mu_expand)
    tmp_distance = reshaped_pred - mu_expand
    distance = tf.norm(tmp_distance, ord=1, axis=1)

    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    ### Calculate l_dist

    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    # Filter out zeros from same cluster subtraction
    eye = tf.eye(num_instances)
    zero = tf.zeros(1, dtype=tf.float32)
    diff_cluster_mask = tf.equal(eye, zero)
    diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])
    mu_diff_bool = tf.boolean_mask(mu_diff, diff_cluster_mask)

    #intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
    #zero_vector = tf.zeros(1, dtype=tf.float32)
    #bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    #mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = tf.norm(mu_diff_bool, ord=1, axis=1)
    mu_norm = tf.subtract(2. * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    def rt_0(): return 0.
    def rt_l_dist(): return l_dist
    l_dist = tf.cond(tf.equal(1, num_instances), rt_0, rt_l_dist)
    
    ### Calculate l_reg
    l_reg = tf.reduce_mean(tf.norm(mu, ord=1, axis=1))

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    return loss, l_var, l_dist, l_reg

class ModelRunner:
    """
    A class to run a specified model on a specified dataset
    """

    def __init__(self, model_conf, data_conf, name, save_root_dir, train_dataset, test_dataset, mode=None):
        """
        Initialize a model runner
        :param model_conf: The pyconf for model
        :param data_conf: The pyconf for dataset
        :param name: The name for model
        :param save_root_dir: The root for saving. Normally it is the root directory where all the models of a specified
        dataset should be saved. Like something "path/ModelNet40-2048". Note that it is not the "root directory of the
        model", such as "path/ModelNet40-2048/PointCNN-X3-L4".
        :param train_dataset: The dataset to train the model
        :param test_dataset: The dataset to test the model
        :param mode: The mode indicates the strategy of whether to reuse the previous training process and continue
        training. Currently we support 3 types of modes:
            1. "new" or None: Do not use the previous result and start from beginning.
            2. "resume": Reuse previous result
            3. "resume-copy": Reuse previous result but make an exact copy.
        Both the "resume" and "resume-copy" will try to find the last result with the same "name" in the "save_root_dir"
        and reuse it. "resume" mode will continue training in the previous directory while "resume-copy" will try to
        create a new one and maintain the original one untouched. Default is None.
        """
        self.model_conf = model_conf
        self.data_conf = data_conf
        self.name = name
        self.save_root_dir = save_root_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self._mode = mode or "new"
        assert self._mode in [_MODE_NEW, _MODE_RESUME, _MODE_RESUME_COPY], \
            "Unrecognized mode=\"{}\". Currently support \"new\", \"resume\" and \"resume-copy\""

    def train(self):
        control_conf = self.model_conf["control"]

        # Transform the dataset is the dataset is classification dataset and
        # the model_conf's last output layer is output-conditional-segmentation
        train_dataset = test_dataset = None
        if self.data_conf["task"] == "classification" and \
                self.model_conf["net"]["layers"][-1]["name"] == "output-segmentation-and-semantic-label":
            layer_conf = self.model_conf["net"]["layers"][-1]
            assert "output_size" in layer_conf, "The dataset is classification dataset " \
                                                "while the model configuration is segmentation. " \
                                                "Cannot find \"output_size\" to transform the " \
                                                "classification dataset to segmentation task"
            #seg_output_size = layer_conf["output_size"]
            # Transform function convert the label with (B, 1) to (B, N) where N is the last layer's point output size
            #transform_func = (lambda points, label: (points, tf.tile(label, (1, seg_output_size))))
            #train_dataset = self.train_dataset.map(transform_func)
            train_dataset = self.train_dataset
            test_dataset = self.test_dataset
            #logger.log("Convert classification to segmentation task with output_size={}".format(seg_output_size))
        else:
            train_dataset, test_dataset = self.train_dataset, self.test_dataset

        # Get the suffix of the directory by iterating the root directory and check which suffix has not been
        # created
        suffix = 0

        # The lambda tries to get the save directory based on the suffix
        def save_dir(suffix_=None):
            suffix_ = suffix_ if suffix_ is not None else suffix
            return path.join(self.save_root_dir, self.name + ("-" + str(suffix_) if suffix_ > 0 else ""))

        # Find the last one that the name has not been occupied
        while path.exists(save_dir()):
            suffix += 1

        # Check mode and create directory
        if self._mode == _MODE_NEW or suffix == 0:
            # We will enter here if the mode is "new" or we cannot find the previous model (suffix == 0)
            if self._mode != _MODE_NEW:
                logger.log("Unable to find the model with name \"{}\" to resume. Try to create new one", color="yellow")
            makedirs(save_dir(), exist_ok=False)
        elif self._mode == _MODE_RESUME:
            # Since we reuse the last one, we decrease it by one and do not need to create directory
            suffix -= 1
        elif self._mode == _MODE_RESUME_COPY:
            # Copy the reused one to the new one
            shutil.copytree(save_dir(suffix - 1), save_dir())
        logger.log("Save in directory: \"{}\"".format(save_dir()), color="blue")

        # Try get the infos and previous train step from the info.txt
        infos = dict()
        infos_file_path = path.join(save_dir(), "info.txt")
        if path.exists(infos_file_path) and path.isfile(infos_file_path):
            with open(path.join(save_dir(), "info.txt")) as f:
                pattern = re.compile(r"(\S+)[\s]?=[\s]*(\S+)")
                for line in f:
                    m = re.match(pattern, line.strip())
                    if m:
                        infos[m.group(1)] = eval(m.group(2))
            logger.log("Info loads, info: {}".format(logger.format(infos)), color="blue")
        else:
            logger.log("Do not find info, maybe it is a newly created model", color="blue")

        # Get the step offset
        # Because we store the "have trained" step, so it needs to increase by 1
        step_offset = infos.get("step", -1) + 1
        logger.log("Set step offset to {}".format(step_offset), color="blue")

        # Get the network
        logger.log("Creating network, train_dataset={}, test_dataset={}".format(self.train_dataset, self.test_dataset))
        net = net_from_config(self.model_conf, self.data_conf)

        # Get the learning_rate and optimizer
        logger.log("Creating learning rate schedule")
        lr_schedule = learning_rate_from_config(control_conf["learning_rate"])
        logger.log("Creating optimizer")
        optimizer = optimizer_from_config(lr_schedule, control_conf["optimizer"])

        # Get the loss
        loss = my_loss_function()

        # Get the metrics
        # We add a logits loss in the metrics since the total loss will have regularization term
        metrics = [my_loss_function()]

        # Get the batch size
        batch_size = control_conf["batch_size"]

        # Get the total step for training
        if "train_epoch" in control_conf:
            train_step = int(math.ceil(control_conf["train_epoch"] * self.data_conf["train"]["size"] / batch_size))
        elif "train_step" in control_conf:
            train_step = control_conf["train_step"]
        else:
            assert False, "Do not set the \"train_step\" or \"train_epoch\" in model configuraiton"

        # Get the validation step
        validation_step = control_conf.get("validation_step", None)
        tensorboard_sync_step = control_conf.get("tensorboard_sync_step", None) or validation_step or 100

        logger.log("Training conf: batch_size={}, train_step={}, validation_step={}, "
                   "tensorboard_sync_step={}".format(batch_size, train_step, validation_step, tensorboard_sync_step))

        # Get the callback
        # Initialize the tensorboard callback, and set the step_offset to make the tensorboard
        # output the correct step
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir(), update_freq=tensorboard_sync_step)
        if hasattr(tensorboard_callback, "_total_batches_seen"):
            tensorboard_callback._total_batches_seen = step_offset
        else:
            logger.log("Unable to set the step offset to the tensorboard, the scalar output may be a messy",
                       color="yellow")

        model_callback = ModelCallback(train_step, validation_step, train_dataset, test_dataset,
                                       batch_size, save_dir(), infos=infos, step_offset=step_offset)

        logger.log("Compile network, loss={}, metrics={}".format(loss, metrics))
        net.compile(optimizer, loss=loss, metrics=metrics)

        logger.log("Summary of the network:")
        net.summary(line_length=240, print_fn=lambda x: logger.log(x, prefix=False))

        logger.log("Begin training")
        net.fit(
            train_dataset,
            verbose=0,
            steps_per_epoch=train_step,
            callbacks=[tensorboard_callback, model_callback],
            shuffle=False  # We do the shuffle ourself
        )