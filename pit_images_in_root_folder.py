import argparse
from PIT_tensor import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='position invariant transform')
    parser.add_argument('--fovx', dest='fovx',
                        help='fovx, the unit is degree',
                        default='0', type=int)
    parser.add_argument('--fovy', dest='fovy',
                        help='fovy, the unit is degree',
                        default='0', type=int)
    parser.add_argument('--root_path', dest='root_path',
                        help='the path to place the image folder',
                        default='./', type=str)
    parser.add_argument('--itp', dest='itp',
                        help='interpolation mode, 1 as nearest, 2 as biliner, 3 as bicubic',
                        default='2', type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)

    for root, dirs, files in os.walk(args.root_path, topdown=True):
        for name in files:
            if not name.endswith('.png') and not name.endswith('.jpg'):
                continue
            im_path = os.path.join(root, name)

            img = Image.open(im_path)
            im = image_to_tensor(img).cuda()  # create input
            width, height = im.shape[3], im.shape[2]
            proj = PIT_module(width, height, fovx = args.fovx / 180 * math.pi, fovy = args.fovy * math.pi / 180)

            im_new = proj.pit(im, interpolation=args.itp, reverse=False)
            im_new = tensor_to_image(im_new)
            im_new.save(im_path)
            img.close()

            print('Image done:', im_path)


