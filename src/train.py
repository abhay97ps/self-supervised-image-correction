import torch
from utils import train_discriminator, train_editor
from pretext import RandomPretextConverter
from editor import editor18
from discriminator import discriminator as disc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(dataset, tasks, batch_size, editor_lr, discriminator_lr, num_of_epochs, l_rec, l_adv, logger):

    unlabelled_data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    num_of_batches = len(unlabelled_data_loader)

    # import the model
    editor = editor18(3).to(device)
    discriminator = disc().to(device)

    # optimizer
    editor_optimizer = torch.optim.Adam(editor.parameters(), lr=editor_lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

    # loss func
    reconstruction_loss = torch.nn.MSELoss()
    adversarial_loss = torch.nn.BCELoss()  # also used for discriminator

    # train the context encoder

    for epoch in range(num_of_epochs):
        for n_batch, (real_data, _) in enumerate(unlabelled_data_loader):
            # generate input data from real data
            input_data = RandomPretextConverter(real_data.clone(), tasks).to(device)
            # 1. Train Discriminator
            # Generate fake data
            fake_data = editor(input_data.clone())
            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(
                discriminator=discriminator,
                optimizer=discriminator_optimizer,
                loss=adversarial_loss,
                real_data=real_data.clone().to(device),
                fake_data=fake_data.to(device),
                device=device
            )

            # 2. Train Editor
            # Generate fake data
            fake_data = editor(input_data.clone())
            # Train E
            e_error = train_editor(
                discriminator=discriminator,
                optimizer=editor_optimizer,
                l_rec=l_rec,
                l_adv=l_adv,
                reconstruction_loss=reconstruction_loss,
                adversarial_loss=adversarial_loss,
                real_data=real_data.clone().to(device),
                fake_data=fake_data.to(device),
                device=device
            )
            # Log error
            logger.log(d_error, e_error, epoch, n_batch, num_of_batches)

            # Display Progress and generate and save test images
            # num_of_batches/x = 10
            if (n_batch) % (num_of_batches//10) == 0:
                logger.display_status(epoch, num_of_epochs, n_batch,
                                      num_of_batches, d_error, e_error, d_pred_real, d_pred_fake)

            # Model Checkpoints
            logger.save_models(editor, discriminator, epoch)
