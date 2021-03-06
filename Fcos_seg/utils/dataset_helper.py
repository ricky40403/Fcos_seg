import math

class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)



def image_list(tensors, size_divisible=0):
    
            
    max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))    
    
    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)
       
    
    shape = (len(tensors),) + max_size
    batch = tensors[0].new(*shape).zero_()

    for img, pad_img in zip(tensors, batch):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    sizes = [img.shape[-2:] for img in tensors]
    
    return ImageList(batch, sizes)



def collate_fn(config):
    def collate_data(batch):
        batch = list(zip(*batch))        
        imgs = image_list(batch[0], config.MODEL.DIV)
        targets = batch[1]
        ids = batch[2]

        return imgs, targets, ids

    return collate_data

    
    