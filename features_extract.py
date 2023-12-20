from data import *
from tqdm import tqdm
import os
from myutil import *


def ibot_inference(img):
    return model(img)


def get_features_and_true_label(inference):
    if os.path.exists(features_path):
        tmp = torch.load(features_path)
        features_numpy = tmp['data']
        labels_numpy = tmp['label']
        del tmp
    else:
        features = []
        labels = []
        for index, (image, label) in enumerate(tq):
            if index == 10: break
            image = image.to(device)
            with torch.no_grad():
                image = processor(images=image, return_tensors="pt", do_rescale=False) if processor != None else image
                features.append(inference(image))
            labels.append(label)

        # save features and labels
        features_numpy = torch.cat(features, dim=0).cpu().numpy()
        labels_numpy = torch.cat(labels).cpu().numpy()
        save_features(features_numpy, labels_numpy)

    return features_numpy, labels_numpy


def save_features(feaures_numpy, labels_numpy):
    torch.save({'data': feaures_numpy, 'label': labels_numpy}, features_path)


if __name__=="__main__":
    setup_seed(100)
    parser = argparse.ArgumentParser(description="test pretraining model")
    config = get_config(parser)

    print(f"use model {config['model']}")
    print(f"image size is {config['features_extract']['image_size']}")

    features_path = os.path.join(config['features_extract']['features_save_dir'],
                                 config['model'] + "_" + config['Dataset'] + "imgSize=" + str(config['features_extract']['image_size']) + ".pt")
    if not os.path.exists("./features"):
        os.mkdir("features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl = get_dataloader_by_name(config['Dataset'], config['features_extract']['image_size'])
    tq = tqdm(dl)

    processor = None
    model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16', pretrained=False)
    checkpoint = torch.load("checkpoint_teacher_vitB16.pth")
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model_inference = ibot_inference

    features_numpy, labels_numpy = get_features_and_true_label(inference=model_inference)



















