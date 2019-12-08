import torch


def readUnlabeledData():
    return 0


def real_data_target(size, device):
    '''
    Tensor containing ones, with shape = size
    '''
    data = torch.ones(size, 1).to(device)
    return data


def fake_data_target(size, device):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = torch.zeros(size, 1).to(device)
    return data


def train_editor(discriminator, optimizer, l_rec, l_adv, reconstruction_loss, adversarial_loss, real_data, fake_data, device):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    # error is reconstruction + adversarial
    error = l_rec * reconstruction_loss(fake_data, real_data) + \
        l_adv * adversarial_loss(prediction, real_data_target(N, device))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


def train_discriminator(discriminator, optimizer, loss, real_data, fake_data, device):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, real_data_target(N, device))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, fake_data_target(N, device))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake
