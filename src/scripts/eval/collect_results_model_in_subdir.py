import SimpleITK as sITK
import numpy as np
import os
import argparse
import subprocess
import uuid
import rootpath
from multiprocessing.pool import ThreadPool
import threading
import surface_distance.metrics as sd

try:
    from scripts.eval import defs
except ImportError:
    import defs


OUTPUT_FILE_KEY = ""
RESULT_SUB_DIR = "out"
ID_IDX = 9
ID_IDX_TCIA = 11


# MEASURES = ['fp', 'fn', 'tp', 'tn', 'fpr', 'fnr',
# 'sensitivity', 'ppv', 'specificity', 'accuracy', 'dice']
MEASURES = ['dice', 'haus_dist']

mutex = threading.Lock()


def get_ground_truth(model, result_file, gt_path, gt_filter):
    file_id = result_file[:ID_IDX_TCIA] if result_file.startswith('TCIA') else result_file[:ID_IDX]
    for gt_file in os.listdir(gt_path):
        print(gt_filter, file_id)
        if gt_filter in gt_file and file_id in gt_file:
            return os.path.join(gt_path, gt_file)
    print("ground truth not found: ", model, result_file)
    return None


def write_header(use_plastimatch):
    if use_plastimatch:
        header = "Model,Checkpoint,File,Organ,dice,95haus_dist,avghaus_dist,surface_dice"
    else:
        header = "Model,Checkpoint,File,Organ," + ",".join(MEASURES)
    print(header)
    return header


def get_plastimatch_results(gt_itk, gt_organ_np, result_itk, result_organ_np,
                            organ, with_sd=False):
    gt2save_itk = sITK.GetImageFromArray(gt_organ_np)
    gt2save_itk.SetSpacing(gt_itk.GetSpacing())
    gt2save_itk.SetOrigin(gt_itk.GetOrigin())
    gt2save_itk.SetDirection(gt_itk.GetDirection())

    if not os.path.exists('tmp'):
        os.makedirs('tmp')

    tmp_gt = os.path.join('tmp', str(uuid.uuid4()) + ".nii.gz")
    sITK.WriteImage(gt2save_itk, tmp_gt)

    res2save_itk = sITK.GetImageFromArray(result_organ_np)
    res2save_itk.SetSpacing(result_itk.GetSpacing())
    res2save_itk.SetOrigin(result_itk.GetOrigin())
    res2save_itk.SetDirection(result_itk.GetDirection())

    tmp_res = os.path.join('tmp', str(uuid.uuid4()) + ".nii.gz")
    sITK.WriteImage(res2save_itk, tmp_res)

    plasti_res_byte = subprocess.check_output(['plastimatch',
                                               'dice', '--all',
                                               tmp_gt, tmp_res])

    plasti_res = plasti_res_byte.decode('utf8')
    os.remove(tmp_gt)
    os.remove(tmp_res)

    dice = ''
    hd95 = ''
    hdavg = ''
    sdice = '0'
    for line in plasti_res.split('\n'):
        if 'DICE:' in line:
            dice = line.rstrip().split(' ')[-1]
        if "Avg average Hausdorff distance (boundary)" in line:
            hdavg = line.rstrip().split(' ')[-1]
        if "Percent (0.95) Hausdorff distance (boundary)" in line:
            hd95 = line.rstrip().split(' ')[-1]

    try:
        float(dice)
        # 724 diagonal distance in pix of a 512*512 img
        if float(hdavg) > 724.0:
            hdavg = "1000"
        if float(hd95) > 724.0:
            hd95 = "1000"
    except ValueError:
        print("Not a float")
        return

    if with_sd:
        surface_distances = sd.compute_surface_distances(
            gt_organ_np.astype(np.bool_), result_organ_np.astype(np.bool_), spacing_mm=gt_itk.GetSpacing())
        tau = defs.tau[organ]
        sdice = sd.compute_surface_dice_at_tolerance(surface_distances, tau)
        sdice = str(sdice)

    res_string = dice + "," + hd95 + "," + hdavg + "," + sdice
    return res_string


def get_measurements(result_file, gt_file, model, checkpoint,
                     use_plastimatch, fileptr, labels, with_sd=False):
    print("eval", result_file, gt_file, model, checkpoint)
    gt_itk = sITK.ReadImage(gt_file)
    gt_np = sITK.GetArrayFromImage(gt_itk)
    result_itk = sITK.ReadImage(result_file)
    result_np = sITK.GetArrayFromImage(result_itk)

    print(gt_itk.GetSpacing(), result_itk.GetSpacing())
    np.testing.assert_almost_equal(gt_itk.GetSpacing(),
                                   result_itk.GetSpacing(),
                                   5, "Spacing dimension does not match")

    result_strings = []

    for key, value in labels.items():
        gt_organ_np = np.zeros_like(gt_np, dtype=np.int8)
        result_organ_np = np.zeros_like(result_np, dtype=np.int8)

        gt_organ_np[gt_np == value] = 1
        result_organ_np[result_np == value] = 1

        if use_plastimatch:
            res_string = get_plastimatch_results(gt_itk,
                                                 gt_organ_np,
                                                 result_itk,
                                                 result_organ_np,
                                                 key,
                                                 with_sd)
        else:
            # uses NiftyNet result measurements
            from niftynet.evaluation.pairwise_measures import PairwiseMeasures

            measures = PairwiseMeasures(result_organ_np,
                                        gt_organ_np,
                                        measures=MEASURES,
                                        pixdim=gt_itk.GetSpacing())
            res_string = measures.to_string()

        organ_result_string = ("%s,%s,%s,%s," + res_string) % \
                              (model, checkpoint,
                               os.path.split(result_file)[1], key)
        result_strings.append(organ_result_string)
        print(threading.currentThread(), organ_result_string)

    mutex.acquire()
    try:
        for result in result_strings:
            fileptr.write(result + "\n")
            fileptr.flush()
    finally:
        mutex.release()


def evaluate(gt_base_path, result_base_path, result_file, model,
             special_checkpoint, use_plastimatch, result_dir, threads,
             num_classes, with_sd=False, checkpoint_filter = ''):
    data = []
    fileptr = open(result_file, 'w')
    header = write_header(use_plastimatch)
    fileptr.write(header + "\n")
    fileptr.flush()

    labels = defs.def_by_class[num_classes]['labels']
    gt_filter = defs.def_by_class[num_classes]['gt_filter']

    for model_dir in os.listdir(result_base_path):

        if model in model_dir:

            full_result_dir = os.path.join(result_base_path,
                                           model_dir, result_dir)
            if not os.path.isdir(full_result_dir):
                continue
            for checkpoint in os.listdir(full_result_dir):

                if not checkpoint_filter in checkpoint:
                    continue

                if special_checkpoint is not None and \
                        special_checkpoint != checkpoint:
                    continue

                full_checkpoint_dir = os.path.join(full_result_dir, checkpoint)
                if os.path.isdir(full_checkpoint_dir):

                    for result_file in os.listdir(full_checkpoint_dir):
                        if OUTPUT_FILE_KEY in result_file:
                            full_result_file = \
                                os.path.join(full_checkpoint_dir, result_file)
                            full_gt_file = get_ground_truth(model_dir,
                                                            result_file,
                                                            gt_base_path,
                                                            gt_filter)
                            if full_gt_file is not None:
                                data.append([full_result_file,
                                             full_gt_file,
                                             model_dir,
                                             checkpoint,
                                             use_plastimatch,
                                             fileptr,
                                             labels,
                                             with_sd])
                else:
                    print("Checkpoint not found: ", full_checkpoint_dir)

    t = ThreadPool(threads)
    t.starmap(get_measurements, data)

    fileptr.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='')

    parser.add_argument('--resultfile',
                        required=True
                        )
    parser.add_argument('--gtdir',
                        required=False,
                        default='./data/miccai/full_dataset_nifti/test'
                        )
    parser.add_argument('--numclasses',
                        required=False,
                        choices=[2, 3, 7, 8],
                        type=int,
                        default=8
                        )
    parser.add_argument('--modeldir',
                        required=False,
                        default='model'
                        )
    parser.add_argument('--model',
                        required=False,
                        default=''
                        )
    parser.add_argument('--checkpoint',
                        required=False,
                        type=str,
                        default=None
                        )
    parser.add_argument('--useplastimatch',
                        action='store_true'
                        )
    parser.add_argument('--withsd',
                        action='store_true'
                        )
    parser.add_argument('--threads', '-t',
                        required=False,
                        type=int,
                        default=1)
    parser.add_argument('--checkpoint_contains',
                        required=False,
                        type=str,
                        default='')

    args = parser.parse_args()
    resfile = os.path.join(rootpath.detect(), args.resultfile) \
        if not os.path.isabs(args.resultfile) else args.resultfile
    modeldir = os.path.join(rootpath.detect(), args.modeldir) \
        if not os.path.isabs(args.modeldir) else args.modeldir
    gtdir = os.path.join(rootpath.detect(), args.gtdir) \
        if not os.path.isabs(args.gtdir) else args.gtdir

    evaluate(gtdir, modeldir, resfile, args.model, args.checkpoint,
             args.useplastimatch, RESULT_SUB_DIR, args.threads,
             args.numclasses, args.withsd, args.checkpoint_contains)
