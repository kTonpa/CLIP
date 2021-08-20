import argparse

import torch
from clip import clip

from clip import loader, tokenize


# model, trainig loop arguments

# argument parsing

parser = argparse.ArgumentParser()


group.add_argument('--model_name', type=str,
                   help='name of CLIP model')

parser.add_argument('--image_text_folder', type=str, required=True,
                    help='path to your folder of images and text for learning the CLIP')

parser.add_argument('--clip_output_file_name', type=str, default = "clip",
                    help='output_file_name')

parser.add_argument('--wandb_name', default='clip_train_transformer',
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')

parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs')

train_group.add_argument('--save_every_n_steps', default = 1000, type = int, help = 'Save a checkpoint every n steps')

train_group.add_argument('--batch_size', default = 4, type = int, help = 'Batch size')

train_group.add_argument('--ga_steps', default = 1, type = int, help = 'Number of steps to accumulate gradients across per each iteration')

train_group.add_argument('--learning_rate', default = 3e-4, type = float, help = 'Learning rate')

train_group.add_argument('--clip_grad_norm', default = 0.5, type = float, help = 'Clip gradient norm')

args = parser.parse_args()

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def create_clip_img_transform(image_width):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    transform = T.Compose([
                    #T.ToPILImage(),
                    T.Resize(image_width),
                    T.CenterCrop((image_width, image_width)),
                    T.ToTensor(),
                    T.Normalize(mean=clip_mean, std=clip_std)
            ])
    return transform



# constants
CLIP_OUTPUT_FILE_NAME = args.clip_output_file_name + ".pt"

CLIP_PATH = args.clip_path

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate
GRAD_CLIP_NORM = args.clip_grad_norm
SAVE_EVERY_N_STEPS = args.save_every_n_steps

# load the dataset and transform

ds = TextImageDataset(
    args.image_text_path,
    text_len=TEXT_SEQ_LEN,
    truncate_captions=args.truncate_captions,
    text_tokenizer=txt_tokenizer,
    transform=ds_transforms,
    shuffle=True,
)

assert len(ds) > 0, 'dataset is empty'
print(f'{len(ds)} image-text pairs found for training')

# Regular DataLoader for image-text-folder datasets
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# load the model in training mode

# Load CLIP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip, norm = load(args.model_name, device=device)

input_res = clip.input_resolution.item()

clip_transform = create_clip_img_transform(input_res)

# optimizer

opt = Adam(get_trainable_params(clip), lr=LEARNING_RATE)

def save_model(path, epoch=0):
    save_obj = {
        'weights': clip.state_dict(),
        'opt_state': opt.state_dict()
    }

    torch.save(save_obj, path)


# training loop

save_model(f'./{CLIP_OUTPUT_FILE_NAME}.pt', epoch=resume_epoch)
# training

steps = 0
for epoch in range(resume_epoch, EPOCHS):
    for i, (texts, images) in enumerate(dl):
        if i % 10 == 0:
            t = time.time()

        texts, images = map(lambda t: t.cuda(), (texts, images))

        loss, text_loss, image_loss = clip(texts, images, return_loss=True)
        loss = loss / ACCUM_STEPS
        loss.backward()

        if (i+1) % ACCUM_STEPS == 0:
            clip_grad_norm_(text2punk.parameters(), GRAD_CLIP_NORM)
            opt.step()
            opt.zero_grad()

        lr = LEARNING_RATE
            
        log = {}

        if i % 10 == 0:
            print(f'epoch - {epoch},', f'step - {i},', f'loss - {loss.item()}')

            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': loss.item(),
                'lr': lr
            }


        if i % 10 == 9:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            print(epoch, i, f'sample_per_sec - {sample_per_sec}')

        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(f'./{CLIP_OUTPUT_FILE_NAME}.pt', epoch=epoch)

        steps += 1
        wandb.log(log)

    # save trained model to wandb as an artifact every epoch's end
    
    model_artifact = wandb.Artifact('finetuned-clip', type='model', metadata=dict(model_config))
    run.log_artifact(model_artifact)


save_model(f'./{CLIP_OUTPUT_FILE_NAME}-final.pt', epoch=epoch)
model_artifact = wandb.Artifact('finetuned-clip', type='model', metadata=dict(model_config))
run.log_artifact(model_artifact)

wandb.finish()
