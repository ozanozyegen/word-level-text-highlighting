import os, subprocess, re
import tensorflow as tf

def auto_gpu_selection(usage_max=0.01, mem_max=0.05, is_tensorflow=True):
    """ Auto set CUDA_VISIBLE_DEVICES for gpu
    Warning: Does not parse correctly in some cases
    :param mem_max: max percentage of GPU utility
    :param usage_max: max percentage of GPU memory
    :return:
    """
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    log = str(subprocess.check_output("nvidia-smi", shell=True)).split(r"\n")[6:-1]
    gpu = 0

    # Maximum of GPUS, 8 is enough for most
    for i in range(8):
        idx = i*3 + 3 # If doesn't work revert +3 to +2
        if idx > log.__len__()-1:
            break
        inf = log[idx].split("|")
        if inf.__len__() < 3:
            break
        usage = int(inf[3].split("%")[0].strip())
        mem_now = int(str(inf[2].split("/")[0]).strip()[:-3])
        mem_all = int(str(inf[2].split("/")[1]).strip()[:-3])
        # print("GPU-%d : Usage:[%d%%]" % (gpu, usage))
        if usage < 100*usage_max and mem_now < mem_max*mem_all:
            if is_tensorflow:
                # Set memory growth
                gpu_devices = tf.config.experimental.list_physical_devices('GPU')
                tf.config.set_visible_devices(gpu_devices[gpu], 'GPU')
                for device in gpu_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            else: # Not tested
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

            print("\nAuto choosing vacant GPU-%d : Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]\n" %
                (gpu, mem_now, mem_all, usage))
            return
        print("GPU-%d is busy: Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]" %
            (gpu, mem_now, mem_all, usage))
        gpu += 1
    print("\nNo vacant GPU, use CPU instead\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""
    def run_command(cmd):
        """Run command, return output as string."""
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
        return output.decode("ascii")

    def list_available_gpus():
        """Returns list of available GPU ids."""
        output = run_command("nvidia-smi -L")
        # lines of the form GPU 0: TITAN X
        gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
        result = []
        for line in output.strip().split("\n"):
            m = gpu_regex.match(line)
            assert m, "Couldnt parse "+line
            result.append(int(m.group("gpu_id")))
        return result

    def gpu_memory_map():
        """Returns map of GPU id to memory allocated on that GPU."""

        output = run_command("nvidia-smi")
        gpu_output = output[output.find("GPU Memory"):]
        # lines of the form
        # |    0      8734    C   python                                       11705MiB |
        memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
        rows = gpu_output.split("\n")
        result = {gpu_id: 0 for gpu_id in list_available_gpus()}
        for row in gpu_output.split("\n"):
            m = memory_regex.search(row)
            if not m:
                continue
            gpu_id = int(m.group("gpu_id"))
            gpu_memory = int(m.group("gpu_memory"))
            result[gpu_id] += gpu_memory
        return result

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def use_gpu(USE_GPU: bool, GPU_ID=None):
    """ Selects GPU for tensorflow
        If USE_GPU set to False, runs on CPU
        If GPU_ID not set, picks gpu with the lowest memory used
    """
    gpus = tf.config.list_physical_devices('GPU')
    print(len(gpus))
    if gpus and USE_GPU:
        if GPU_ID is None:
            GPU_ID = pick_gpu_lowest_memory()
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[GPU_ID], 'GPU')
            for gpu in gpus: # Limit Memory Growth for each GPU
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    else:
        tf.config.set_visible_devices([], 'GPU')
