import torch
from utils import readUnlabeledData, train_editor, train_discriminator
from editor import editor18
from discriminator import discriminator as disc
from logger import Logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(batch_size, editor_lr, discriminator_lr, num_of_epochs, l_rec, l_adv, logger):
    # load data
    data = readUnlabeledData()
    unlabelled_data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True)
    num_of_batches = len(unlabelled_data_loader)

    # import the model
    editor = editor18(3).to(device)
    discriminator = disc().to(device)

    # optimizer
    editor_optimizer = torch.optim.Adam(editor.parameters(), lr=editor_lr)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=discriminator_lr)

    # loss func
    reconstruction_loss = torch.nn.MSELoss()
    adversarial_loss = torch.nn.BCELoss()  # also used for discriminator

    # train the context encoder

    for epoch in range(num_of_epochs):
        for n_batch, real_data in enumerate(unlabelled_data_loader):

            # 1. Train Discriminator
            # Generate fake data
            fake_data = editor(real_data).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(
                discriminator=discriminator,
                optimizer=discriminator_optimizer,
                loss=adversarial_loss,
                real_data=real_data,
                fake_data=fake_data,
                device=device
            )

            # 2. Train Editor
            # Generate fake data
            fake_data = editor(real_data)
            # Train E
            e_error = train_editor(
                discriminator=discriminator,
                optimizer=editor_optimizer,
                l_rec=l_rec,
                l_adv=l_adv,
                reconstruction_loss=reconstruction_loss,
                adversarial_loss=adversarial_loss,
                real_data=real_data,
                fake_data=fake_data,
                device=device
            )
            # Log error
            logger.log(d_error, e_error, epoch, n_batch, num_of_batches)

            # Display Progress
            logger.display_status(epoch, num_of_epochs, n_batch,
                                num_of_batches, d_error, e_error, d_pred_real, d_pred_fake)

            # Model Checkpoints
            logger.save_models(editor, discriminator, epoch)
