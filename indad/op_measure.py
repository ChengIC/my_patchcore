import os
from visual_patchcore import visPatchCore


if __name__ == "__main__":
    test_imgs_folder = './datasets/full_body/test/objs'
    annotation_folder = './datasets/full_body/Annotations'

    for roots, dirs, files in os.walk('./results_example'):
        if dirs == ['imgs', 'data']:
            result_dirs = [os.path.join(roots,'data')]
            av1 = visPatchCore (all_json_dirs=result_dirs,
                                test_imgs_folder=test_imgs_folder,
                                annotation_folder=annotation_folder,
                                write_image=True,
                                write_result=True,
                                TimeStamp=None)
            av1.vis_result()

