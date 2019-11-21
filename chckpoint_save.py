with tf.name_scope('prev_checkpoint_loader'):
    if os.path.isfile(config.checkpoint_file_check):
        ckpts = tf.train.latest_checkpoint(config.ckpt_load_folder)
        vars_in_checkpoint = tf.train.list_variables(ckpts)
        variables = tf.contrib.slim.get_variables_to_restore()
        ckpt_var_name = []
        ckpt_var_shape = {}
        for el in vars_in_checkpoint:
            ckpt_var_name.append(el[0])
            ckpt_var_shape[el[0]] = el[1]
        var_list = [v for v in variables if v.name.split(':')[0] in ckpt_var_name]
        var_list = [v for v in var_list if list(v.shape) == ckpt_var_shape[v.name.split(':')[0]]]
        self.prev_checkpoint_loader = tf.train.Saver(var_list=var_list)