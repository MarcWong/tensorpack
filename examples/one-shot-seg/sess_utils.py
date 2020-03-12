# Author: Tao Hu <taohu620@gmail.com>
import numpy as np
import tensorflow as tf
from tensorpack.utils import logger
import six,copy
from tensorpack.tfutils.common import get_op_tensor_name
from tensorpack.tfutils.varmanip import (SessionUpdate, get_savename_from_varname,
                       is_training_name, get_checkpoint_path)

from tensorpack.tfutils.sessinit import DictRestore,SaverRestore, SessionInit, MismatchLogger,CheckpointReaderAdapter, ChainInit

def my_get_model_loader(filename):
    """
    Get a corresponding model loader by looking at the file name.

    Returns:
        SessInit: either a :class:`DictRestore` (if name ends with 'npy/npz') or
        :class:`SaverRestore` (otherwise).
    """
    if filename.endswith('.npy'):
        assert tf.gfile.Exists(filename), filename
        return DictRestore(np.load(filename, encoding='latin1').item())
    elif filename.endswith('.npz'):
        assert tf.gfile.Exists(filename), filename
        obj = np.load(filename)
        return MyDictRestore(dict(obj))
    else:
        return ChainInit([SaverRestore(filename,prefix='support'),SaverRestore(filename,prefix='query')])

class MyDictRestore(SessionInit):
    """
    Restore variables from a dictionary.
    """

    def __init__(self, variable_dict):
        """
        Args:
            variable_dict (dict): a dict of {name: value}
        """
        assert isinstance(variable_dict, dict), type(variable_dict)
        # use varname (with :0) for consistency
        self._prms = {get_op_tensor_name(n)[1]: v for n, v in six.iteritems(variable_dict)}

    def _run_init(self, sess):
        total_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for branch in ['support','query']:
            variables = [v for v in total_variables if branch in v.name]
            variable_names = set([k.name.replace("{}/".format(branch),"") for k in variables])
            param_names = set(six.iterkeys(self._prms))

            intersect = variable_names & param_names

            logger.info("Variables in branch-{} to restore from dict: {}".format(branch, ', '.join(map(str, intersect))))

            mismatch = MismatchLogger('graph', 'dict')
            for k in sorted(variable_names - param_names):
                if not is_training_name(k):
                    mismatch.add(k)
            mismatch.log()
            mismatch = MismatchLogger('dict', 'graph')
            for k in sorted(param_names - variable_names):
                mismatch.add(k)
            mismatch.log()

            upd = SessionUpdate(sess, [v for v in variables if v.name.replace("{}/".format(branch),"") in intersect])
            logger.info("Restoring branch-{}  from dict ...".format(branch))
            upd.update({"{}/{}".format(branch,name): value for name, value in six.iteritems(self._prms) if name.replace("{}/".format(branch),"") in intersect})


class MySaverRestore(SessionInit):
    """
    Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    """
    def __init__(self, model_path, prefixs, ignore=[]):
        """
        Args:
            model_path (str): a model name (model-xxxx) or a ``checkpoint`` file.
            prefix (str): during restore, add a ``prefix/`` for every variable in this checkpoint.
            ignore (list[str]): list of tensor names that should be ignored during loading, e.g. learning-rate
        """
        if model_path.endswith('.npy') or model_path.endswith('.npz'):
            logger.warn("SaverRestore expect a TF checkpoint, but got a model path '{}'.".format(model_path) +
                        " To load from a dict, use 'DictRestore'.")
        model_path = get_checkpoint_path(model_path)
        self.path = model_path  # attribute used by AutoResumeTrainConfig!
        self.prefixs = prefixs # a list
        self.ignore = [i if i.endswith(':0') else i + ':0' for i in ignore]

    def _setup_graph(self):
        dic = self._get_restore_dict()
        self.saver = tf.train.Saver(var_list=dic, name=str(id(dic)))

    def _run_init(self, sess):
        logger.info("Restoring checkpoint from {} ...".format(self.path))
        self.saver.restore(sess, self.path)

    @staticmethod
    def _read_checkpoint_vars(model_path):
        """ return a set of strings """
        reader = tf.train.NewCheckpointReader(model_path)
        reader = CheckpointReaderAdapter(reader)    # use an adapter to standardize the name
        ckpt_vars = reader.get_variable_to_shape_map().keys()
        return reader, set(ckpt_vars)

    def _match_vars(self, func):
        reader, chkpt_vars = SaverRestore._read_checkpoint_vars(self.path)
        graph_vars = tf.global_variables()


        for prefix in self.prefixs:
            chkpt_vars_used = set()
            logger.info("current processing branch: {}".format(prefix))
            mismatch = MismatchLogger('graph', 'checkpoint')
            for v in graph_vars:
                name = get_savename_from_varname(v.name, varname_prefix=prefix)  # if contains prefix, just remove it
                if name in self.ignore and reader.has_tensor(name):
                    logger.info("Variable {} in the graph will not be loaded from the checkpoint!".format(name))
                else:
                    if reader.has_tensor(name):
                        func(reader, name, v)
                        chkpt_vars_used.add(name)
                        logger.info("success load {}".format(v.name))
            """
                    else:
                        vname = v.op.name
                        if not is_training_name(vname):
                            mismatch.add(vname)
            mismatch.log()
            mismatch = MismatchLogger('checkpoint', 'graph')
            if len(chkpt_vars_used) < len(chkpt_vars):
                unused = chkpt_vars - chkpt_vars_used
                for name in sorted(unused):
                    if not is_training_name(name):
                        mismatch.add(name)
            mismatch.log()
            """

    def _get_restore_dict(self):
        var_dict = {}

        def f(reader, name, v):
            name = reader.get_real_name(name)
            if name in var_dict:
                pass
            assert name not in var_dict, "Restore conflict: {} and {}".format(v.name, var_dict[name].name)
            var_dict[name] = v
        self._match_vars(f)
        return var_dict
