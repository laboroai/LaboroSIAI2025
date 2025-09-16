import os
import tqdm
import numpy as np
import pandas as pd
import struct
from PIL import Image
from typing import Literal

def read_and_write_dataset(prefix: Literal['train', 'test'], output_dir: str) -> None:
    print(f"Reading {prefix} dataset...")
    images_filename = f'{prefix}-images-idx3-ubyte'
    labels_filename = f'{prefix}-labels-idx1-ubyte'
    with open(os.path.join('dataset', images_filename), 'rb') as f_image:
        # Read meta data for image
        magic_number = f_image.read(4)
        magic_number = struct.unpack('>i', magic_number)[0]

        number_of_images = f_image.read(4)
        number_of_images = struct.unpack('>i', number_of_images)[0]

        number_of_rows = f_image.read(4)
        number_of_rows = struct.unpack('>i', number_of_rows)[0]

        number_of_columns = f_image.read(4)
        number_of_columns = struct.unpack('>i', number_of_columns)[0]

        bytes_per_image = number_of_rows * number_of_columns
        with open(os.path.join('dataset', labels_filename), 'rb') as f_label:
            # Read meta data for label
            magic_number = f_label.read(4)
            magic_number = struct.unpack('>i', magic_number)[0]

            number_of_labels = f_label.read(4)
            number_of_labels = struct.unpack('>i', number_of_labels)[0]
            if number_of_images != number_of_labels:
                raise ValueError("Number of images and labels do not match.")
            df = {'filename': [], 'width': [], 'height': [], 'label': []}
            for i in tqdm.tqdm(range(number_of_images), desc=f"Processing {prefix} dataset"):
                raw_img = f_image.read(bytes_per_image)
                format = '%dB' % bytes_per_image
                lin_img = struct.unpack(format, raw_img)
                np_ary = np.asarray(lin_img).astype('uint8')
                np_ary = np.reshape(np_ary, (number_of_rows, number_of_columns), order='C')

                filename = f"{prefix}_{i:07d}.png"
                output_path = os.path.join(output_dir, filename)

                pil_img = Image.fromarray(np_ary)
                pil_img.save(output_path)

                label_byte = f_label.read(1)
                label_int = int.from_bytes(label_byte, byteorder='big')
                
                df['filename'].append(filename)
                df['width'].append(pil_img.width)
                df['height'].append(pil_img.height)
                df['label'].append(label_int)

            df = pd.DataFrame(df)
            df.to_csv(os.path.join(output_dir, "labels.csv"), index=False)

    print(f"{prefix} dataset processed and images saved as PNG files.")

if __name__ == "__main__":
    read_and_write_dataset('train', os.path.join('dataset', 'train'))
    read_and_write_dataset('test', os.path.join('dataset', 'test'))
