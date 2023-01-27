from gen_videos import generate_different_images_split, load_G, load_ws
G_base, ws_list = load_G(), load_ws()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--factor', type=float, required=True)
args = parser.parse_args()

#generate_different_images_split(list(range(len(ws_list))), ws_list, G_base, output_resolution=4096, output_format="png", grid=4, extra="p000000/", mov_scale=0.25, num_pixels=1024)
print(f'Using factor {args.factor}, 1/{1024 * args.factor:10f}')
generate_different_images_split(list(range(len(ws_list))), ws_list, G_base, output_resolution=4096, output_format="png", grid=4, extra="p000000/", mov_scale=1, num_pixels=int(1024 * args.factor))