# image_encoder
Encode images into TFRecord data format

# Example

```python
import image_encoder as ie

ie.encode(input_dirs = ['./Train_0', './Train_1', './Train_2'], 
          output_dir = './record', 
          image_length = 28, 
          image_type='png', 
          image_per_shard = 10000)

```

## parameters

- input_dirs: directory list of images, it is supposed that images are in a directory whose name is thier label, for example: ~/Train_0/0000/f8z7sa9kjs81jhskzmn.png
- output_dir: the directory of output TFRecord files
- image_length: for convinience, images are resized to (image_length * image_length) before encoded
- image_per_shard: the record file will be splitted into shards according to this parameter
