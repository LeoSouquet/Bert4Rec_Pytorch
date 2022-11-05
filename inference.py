from common_utils.model_inference import Bert4Rec_inference
import common_utils.metadata as metadata


if __name__ == "__main__":

    model = Bert4Rec_inference(metadata) 

    seg_1 = list(range(1, 2))
    seg_2 = list(range(1, 3))

    print(f"Input Sequence : {seg_1}")
    print(f"Input Sequence : {seg_2}")
    print("---")

    top_k = 15

    print(f" Inference : {model.detect([seg_1,seg_2], top_k)}")
    print(f" Inference : {model.detect([seg_2], top_k)}")
        