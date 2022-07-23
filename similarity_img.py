from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os


def similarity(item):
    model = SentenceTransformer('clip-ViT-B-32')
    image_names = list(glob.glob('.\\match_data\\*.png'))  # 圓圖資料夾
    image_names.append(item)
    # print(image_names)
    # print("Images:", len(image_names))
    encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128,
                                 convert_to_tensor=True, show_progress_bar=True)

    processed_images = util.paraphrase_mining_embeddings(encoded_image)
    NUM_SIMILAR_IMAGES = 10

    # duplicates = [image for image in processed_images if image[0] >= 0.999]
    #
    # for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
    #     print("\nScore: {:.3f}%".format(score * 100))
    #     print(image_names[image_id1])
    #     print(image_names[image_id2])

    threshold = 0.99
    near_duplicates = [image for image in processed_images if image[0] < threshold]

    source = {}
    for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
        print("\nScore: {:.3f}%".format(score * 100))
        print(image_names[image_id1])
        print(image_names[image_id2])
        if image_names[image_id2] == item:
            source.update({image_names[image_id1]:score * 100})
    print(source)
    return source

similarity('.\\1.png')
