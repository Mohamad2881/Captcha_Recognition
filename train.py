import torch
from model import CaptchaModel
from utlis import *



BATCH_SIZE = 8
DEVICE = "cuda"
EPOCHS = 100
PATH = './saved_models/model_1.pth'

def run_training():
    train_dataset, test_dataset, test_labels_orig, le = get_datasets()
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = CaptchaModel(len(le.classes_))
    # model = ResNet(len(le.classes_))
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, mode='max', patience=3, verbose=True)


    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_dataloader, optimizer)
        valid_loss, valid_preds = eval_one_epoch(model, test_dataloader)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss
        }, PATH)

        # print(valid_preds[0].shape)  # ([72, 8, 20]) seqences T, bs, classes
        valid_preds_decoded = []

        chars = list(le.classes_)
        chars = ''.join(chars)
        beam_decoded = []

        for v in valid_preds:
            valid_preds_decoded.extend(decode_preds(v, le))
            # beam_decoded.extend(beam_decoder(v, chars))

        print(list(zip(test_labels_orig, valid_preds_decoded))[0:6])
        print(f'Epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}')
        scheduler.step(valid_loss)


if __name__ == '__main__':
    run_training()